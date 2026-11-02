import uuid
from decimal import Decimal

import pytest
from fakes import FakeProvider, call, reply

from ledger_agent.agent.graph import audit, build_graph
from ledger_agent.agent.state import Deps, Limits
from ledger_agent.db import connect_admin
from ledger_agent.erp.faults import build_scenario
from ledger_agent.erp.load import load
from ledger_agent.llm.client import ModelClient
from ledger_agent.models import FaultType, ScenarioSpec
from ledger_agent.report import write_report
from ledger_evals.bad_fixes import FIRST_OPEN, _true_fix

TABLES = ["GLPERD", "GLCOA", "GLJRNH", "GLJRNL", "APVEND", "APINVH", "APPAYM", "SYSDICT"]


@pytest.fixture(scope="module")
def scenario():
    spec = ScenarioSpec(id="graph", split="dev", seed=11,
                        faults={FaultType.UNBALANCED_JOURNAL: 1}, decoys=[])
    dataset, key = build_scenario(spec)
    name = f"graph_{uuid.uuid4().hex[:12]}"
    load(name, dataset)
    try:
        yield name, key.faults[0]
    finally:
        with connect_admin("postgres") as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
def db(scenario):
    return scenario[0]


@pytest.fixture
def fault(scenario):
    return scenario[1]


def evidence_sql(fault):
    jrnl = next(r.key for r in fault.rows if r.table == "GLJRNH")
    return f'SELECT "JRNL_NO" FROM "GLJRNH" WHERE "JRNL_NO" = {jrnl}'


def finding_item(fault, type="unbalanced_journal"):
    jrnl = next(r.key for r in fault.rows if r.table == "GLJRNH")
    return {"type": type, "summary": "journal does not balance",
            "affected_rows": [{"table": "GLJRNH", "key": jrnl}],
            "evidence_sql": evidence_sql(fault)}


def true_args(fault):
    return _true_fix(fault, FIRST_OPEN).model_dump(mode="json")


def bad_args(fault):
    args = true_args(fault)
    line = args["lines"][0]  # a wrong amount: the journal stays unbalanced
    line["dr_amt" if Decimal(line["dr_amt"]) else "cr_amt"] = "1.00"
    return args


def discovery(*items, tokens=0):
    return [
        reply(call("submit_schema_notes", "n1", notes="notes"), tokens=tokens),
        reply(call("submit_findings", "f1", findings=list(items)), tokens=tokens),
    ]


def propose_reply(args, call_id="p1"):
    return reply(call("submit_correction", call_id, **args))


def run(db, tmp_path, script, limits=None):
    provider = FakeProvider(script)
    client = ModelClient(provider, "live", tmp_path)
    return audit(db, Deps(client, limits or Limits())), provider


def test_passing_fix_ends_verified(db, fault, tmp_path):
    result, provider = run(db, tmp_path, [
        *discovery(finding_item(fault)), propose_reply(true_args(fault))])
    [outcome] = result.state["outcomes"]
    assert outcome.status == "verified" and outcome.attempts == 1
    assert outcome.rendered_sql and "INSERT INTO" in outcome.rendered_sql
    assert [e.node for e in result.trace] == [
        "explore", "hunt", "confirm", "pick", "propose", "verify", "pick"]
    assert result.queries == 1


def test_failed_fix_retries_with_reason(db, fault, tmp_path):
    result, provider = run(db, tmp_path, [
        *discovery(finding_item(fault)),
        propose_reply(bad_args(fault)), propose_reply(true_args(fault))])
    [outcome] = result.state["outcomes"]
    assert outcome.status == "verified" and outcome.attempts == 2
    assert "balance" in provider.requests[-1].messages[-1]["content"]


def test_three_failures_end_needs_human(db, fault, tmp_path):
    result, provider = run(db, tmp_path, [
        *discovery(finding_item(fault)),
        *[propose_reply(bad_args(fault)) for _ in range(3)]])
    [outcome] = result.state["outcomes"]
    assert outcome.status == "needs_human" and outcome.attempts == 3
    assert "balance" in outcome.reason
    assert len(provider.requests) == 5


def test_malformed_proposal_retries_once_then_needs_human(db, fault, tmp_path):
    result, provider = run(db, tmp_path, [
        *discovery(finding_item(fault)),
        reply(call("submit_correction", "p1", reasoning="oops")),  # fields missing
        reply(text="no tool call")])
    [outcome] = result.state["outcomes"]
    assert outcome.status == "needs_human" and outcome.attempts == 0
    assert len(provider.requests) == 4
    retry = provider.requests[-1].messages[-1]["content"]
    assert any("lines" in str(block) for block in retry)  # the validation error went back
    assert provider.requests[-1].tools[0]["name"] == "submit_correction"


def test_malformed_retry_does_not_use_a_fix_attempt(db, fault, tmp_path):
    result, _ = run(db, tmp_path, [
        *discovery(finding_item(fault)),
        reply(text="no tool call"), propose_reply(true_args(fault))])
    [outcome] = result.state["outcomes"]
    assert outcome.status == "verified" and outcome.attempts == 1


def test_other_type_goes_to_needs_human_without_propose(db, fault, tmp_path):
    result, provider = run(db, tmp_path, discovery(finding_item(fault, "other")))
    [outcome] = result.state["outcomes"]
    assert outcome.status == "needs_human" and outcome.attempts == 0
    assert len(provider.requests) == 2


def test_propose_prompt_is_deterministic_and_marks_data_untrusted(db, fault, tmp_path):
    prompts = []
    for _ in range(2):
        _, provider = run(db, tmp_path, [
            *discovery(finding_item(fault)), propose_reply(true_args(fault))])
        prompts.append(provider.requests[-1].messages[0]["content"])
    assert prompts[0] == prompts[1]
    assert db not in prompts[0] and "untrusted" in prompts[0]
    assert evidence_sql(fault) in prompts[0]
    assert [t["name"] for t in provider.requests[-1].tools] == ["submit_correction"]


def test_token_limit_stops_and_reports_partial(db, fault, tmp_path):
    second = finding_item(fault, "other") | {"summary": "second"}
    result, provider = run(
        db, tmp_path, discovery(finding_item(fault), second, tokens=100),
        Limits(tokens=150))
    assert "tokens" in result.state["limits_hit"]
    outcomes = result.state["outcomes"]
    assert [o.status for o in outcomes] == ["needs_human"] * len(outcomes) and outcomes
    assert all(o.reason == "token limit reached" for o in outcomes)
    assert len(provider.requests) == 2
    write_report(result, tmp_path / "out")
    assert "token limit reached" in (tmp_path / "out" / "report.md").read_text()


def test_no_findings_completes_cleanly(db, tmp_path):
    result, _ = run(db, tmp_path, discovery())
    write_report(result, tmp_path / "out")
    assert "No discrepancies found" in (tmp_path / "out" / "report.md").read_text()
    assert result.state["outcomes"] == []


def test_graph_leaves_database_unchanged(db, fault, tmp_path):
    def snapshot():
        with connect_admin(db) as c:
            return [c.execute(
                f"SELECT count(*), md5(coalesce(string_agg(t::text, ',' ORDER BY t::text), '')) "
                f'FROM "{table}" t').fetchone() for table in TABLES]

    before = snapshot()
    run(db, tmp_path, [*discovery(finding_item(fault)), propose_reply(true_args(fault))])
    assert snapshot() == before


def test_audit_can_run_twice_on_one_deps_with_fresh_trace(db, tmp_path):
    provider = FakeProvider([*discovery(), *discovery()])
    deps = Deps(ModelClient(provider, "live", tmp_path))
    first = audit(db, deps)
    second = audit(db, deps)
    assert len(first.trace) == len(second.trace)


def test_build_graph_compiles(db, tmp_path):
    deps = Deps(ModelClient(FakeProvider([]), "live", tmp_path))
    assert build_graph(deps) is not None
