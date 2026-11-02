import uuid
import psycopg
from decimal import Decimal

import pytest
from fakes import FakeProvider, call, reply

from ledger_agent.agent.graph import audit, build_graph
from ledger_agent.agent.state import Deps, Limits
from ledger_agent import cli
from ledger_agent.db import DatabaseUnavailable, connect_admin
from ledger_agent.erp.faults import build_scenario
from ledger_agent.erp.load import load
from ledger_agent.llm.client import ModelClient, ModelRequest, ModelResponse
from ledger_agent.models import Correction, FaultType, ScenarioSpec
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


def test_rejected_corrections_are_kept_in_order(db, fault, tmp_path):
    bad = bad_args(fault)
    result, _ = run(db, tmp_path, [
        *discovery(finding_item(fault)), propose_reply(bad), propose_reply(true_args(fault))])
    assert [c.model_dump(mode="json") for c in result.state["rejected"]] == [
        Correction.model_validate(bad).model_dump(mode="json")]
    gave_up, _ = run(db, tmp_path, [
        *discovery(finding_item(fault)), *[propose_reply(bad) for _ in range(3)]])
    assert len(gave_up.state["rejected"]) == 3


def test_malformed_proposals_are_kept_even_when_a_retry_succeeds(db, fault, tmp_path):
    raw = {"lines": [{"acct_cd": "ZZ99"}], "reasoning": "oops"}
    result, _ = run(db, tmp_path, [
        *discovery(finding_item(fault)),
        reply(call("submit_correction", "p1", **raw)), propose_reply(true_args(fault))])
    assert result.state["malformed"] == [raw]
    assert result.state["outcomes"][0].status == "verified"


def test_malformed_proposals_are_kept_when_the_retry_fails(db, fault, tmp_path):
    raw = {"lines": "ZZ99", "reasoning": "oops"}
    result, _ = run(db, tmp_path, [
        *discovery(finding_item(fault)),
        reply(call("submit_correction", "p1", **raw)), reply(call("submit_correction", "p2", **raw))])
    assert result.state["malformed"] == [raw, raw]


def test_malformed_proposal_is_kept_when_the_token_limit_stops_the_retry(db, fault, tmp_path):
    raw = {"lines": "ZZ99", "reasoning": "oops"}
    result, _ = run(db, tmp_path, [
        *discovery(finding_item(fault)),
        reply(call("submit_correction", "p1", **raw), tokens=100)], Limits(tokens=100))
    assert result.state["malformed"] == [raw]
    assert "tokens" in result.state["limits_hit"]


def test_audit_splits_tokens_for_this_run_only(db, tmp_path):
    earlier = ModelResponse(text="", input_tokens=3, output_tokens=2)
    notes, findings = discovery(tokens=5)
    findings.output_tokens = 4
    deps = Deps(ModelClient(FakeProvider([earlier, notes, findings]), "live", tmp_path))
    deps.client.complete(ModelRequest("m", "s", [], []))
    result = audit(db, deps)
    assert (result.tokens, result.input_tokens, result.output_tokens) == (14, 10, 4)


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


@pytest.fixture(scope="module")
def two_faults():
    spec = ScenarioSpec(id="graph2", split="dev", seed=12,
                        faults={FaultType.UNBALANCED_JOURNAL: 2}, decoys=[])
    dataset, key = build_scenario(spec)
    name = f"graph2_{uuid.uuid4().hex[:12]}"
    load(name, dataset)
    try:
        yield name, key.faults
    finally:
        with connect_admin("postgres") as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def test_second_finding_starts_fresh_after_a_retry(two_faults, tmp_path):
    db, (one, two) = two_faults
    result, provider = run(db, tmp_path, [
        *discovery(finding_item(one), finding_item(two)),
        propose_reply(bad_args(one)), propose_reply(true_args(one)),
        propose_reply(true_args(two))])
    assert [(o.finding_id, o.status, o.attempts) for o in result.state["outcomes"]] == [
        ("F001", "verified", 2), ("F002", "verified", 1)]
    assert "rejected" not in provider.requests[-1].messages[-1]["content"]


def test_second_finding_starts_fresh_after_giving_up_on_the_first(two_faults, tmp_path):
    db, (one, two) = two_faults
    result, provider = run(db, tmp_path, [
        *discovery(finding_item(one), finding_item(two)),
        *[propose_reply(bad_args(one)) for _ in range(3)], propose_reply(true_args(two))])
    assert [(o.finding_id, o.status, o.attempts) for o in result.state["outcomes"]] == [
        ("F001", "needs_human", 3), ("F002", "verified", 1)]
    assert "rejected" not in provider.requests[-1].messages[-1]["content"]


# Database loss ends the run ------------------------------------------------------------


def cli_run(db, tmp_path, monkeypatch, script):
    provider = FakeProvider(script)
    monkeypatch.setattr(cli, "client_from_env",
                        lambda ns: ModelClient(provider, "live", tmp_path / "rec"))
    out = tmp_path / "out"
    return cli.main(["audit", "--db", db, "--out", str(out)]), out


def assert_failed_cleanly(code, out, capsys):
    err = capsys.readouterr().err
    assert code == 1 and "Traceback" not in err and len(err.strip().splitlines()) == 1
    assert not (out / "report.md").exists()


def test_database_lost_after_hunt_ends_run_with_exit_1(db, fault, tmp_path, monkeypatch, capsys):
    from ledger_agent.agent import graph
    from ledger_agent.tools import sql

    real_hunt = graph.hunt

    def hunt_then_lose_database(state, deps):
        update = real_hunt(state, deps)

        def refuse(dbname):
            raise DatabaseUnavailable("cannot connect to database")

        monkeypatch.setattr(sql, "connect_ro", refuse)
        return update

    monkeypatch.setattr(graph, "hunt", hunt_then_lose_database)
    code, out = cli_run(db, tmp_path, monkeypatch, discovery(finding_item(fault)))
    assert_failed_cleanly(code, out, capsys)


def test_connection_lost_in_verify_trial_ends_run_with_exit_1(
        db, fault, tmp_path, monkeypatch, capsys):
    from ledger_agent.verifier import verify as verify_module

    def dead(trial):
        raise psycopg.OperationalError("server closed the connection unexpectedly")

    monkeypatch.setattr(verify_module, "_snapshot", dead)
    code, out = cli_run(db, tmp_path, monkeypatch,
                        [*discovery(finding_item(fault)), propose_reply(true_args(fault))])
    assert_failed_cleanly(code, out, capsys)


def test_query_level_errors_stay_tool_errors(db):
    from ledger_agent.tools.sql import run_query

    assert not run_query(db, 'SELECT "NO_SUCH_COLUMN" FROM "GLJRNH"').ok
    assert not run_query(db, "SELECT pg_sleep(1)", timeout_ms=1).ok


def test_run_query_raises_when_database_unreachable(db, monkeypatch):
    from ledger_agent.tools.sql import run_query

    monkeypatch.setenv("LEDGER_DB_PORT", "1")
    with pytest.raises(DatabaseUnavailable):
        run_query(db, "SELECT 1")
