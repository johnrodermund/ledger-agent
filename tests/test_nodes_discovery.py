import json
import re
import uuid

import pytest
from fakes import FakeProvider, call, reply

from ledger_agent.agent.loop import run_tool_loop
from ledger_agent.agent.nodes import Deps, confirm, explore, hunt
from ledger_agent.agent.prompts import (
    EXPLORE_SYSTEM,
    HUNT_SYSTEM,
    RUN_SQL_TOOL,
    SUBMIT_FINDINGS_TOOL,
    SUBMIT_SCHEMA_NOTES_TOOL,
)
from ledger_agent.agent.state import Limits
from ledger_agent.db import connect_admin
from ledger_agent.erp.generate import generate_clean
from ledger_agent.erp.load import load
from ledger_agent.llm.client import ModelClient, ModelResponse
from ledger_agent.models import Finding, RowRef

GOOD_SQL = 'SELECT "JRNL_NO" FROM "GLJRNH" ORDER BY 1 LIMIT 3'
EMPTY_SQL = 'SELECT "JRNL_NO" FROM "GLJRNH" WHERE "JRNL_NO" < 0'


@pytest.fixture(scope="module")
def db():
    name = f"nodes_{uuid.uuid4().hex[:12]}"
    load(name, generate_clean(7))
    try:
        yield name
    finally:
        with connect_admin("postgres") as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def make_client(tmp_path, script):
    provider = FakeProvider(script)
    return ModelClient(provider, "live", tmp_path), provider


def sql_call(sql, call_id="q1"):
    return call("run_sql", call_id, sql=sql)


def notes_call(text="notes", call_id="s1"):
    return call("submit_schema_notes", call_id, notes=text)


def finding(type="duplicate_invoice", rows=(("GLJRNH", 1),), sql=GOOD_SQL, summary="s"):
    return {
        "type": type,
        "summary": summary,
        "affected_rows": [{"table": t, "key": k} for t, k in rows],
        "evidence_sql": sql,
    }


def findings_call(*items, call_id="f1"):
    return call("submit_findings", call_id, findings=list(items))


def tool_results(req):
    """The tool_result blocks in the last user message of a request."""
    return [b for b in req.messages[-1]["content"] if b["type"] == "tool_result"]


def loop(client, db, query_limit=5, **kw):
    return run_tool_loop(
        client, "sys", [{"role": "user", "content": "go"}], query_limit,
        SUBMIT_SCHEMA_NOTES_TOOL, db, **kw,
    )


def test_loop_returns_sql_errors_to_model_and_counts_them(db, tmp_path):
    client, provider = make_client(tmp_path, [
        reply(sql_call("DELETE FROM \"GLJRNL\"")),
        reply(notes_call()),
    ])
    result = loop(client, db)
    assert result.submitted == {"notes": "notes"} and result.queries_used == 1
    (res,) = tool_results(provider.requests[1])
    body = json.loads(res["content"])
    assert res["is_error"] and body["error"] and body["truncated"] is False


def test_loop_forces_submit_at_query_limit(db, tmp_path):
    client, provider = make_client(tmp_path, [
        reply(sql_call(GOOD_SQL, "a")),
        reply(sql_call(GOOD_SQL, "b")),
        reply(notes_call()),
    ])
    result = loop(client, db, query_limit=2)
    assert result.limit_hit and result.queries_used == 2
    assert result.submitted == {"notes": "notes"}
    last = provider.requests[2]
    assert [t["name"] for t in last.tools] == ["submit_schema_notes"]
    assert last.tool_choice == "submit_schema_notes"
    assert [t["name"] for t in provider.requests[0].tools] == ["run_sql", "submit_schema_notes"]


def test_loop_answers_every_call_and_limits_a_batch(db, tmp_path):
    client, provider = make_client(tmp_path, [
        reply(sql_call(GOOD_SQL, "a"), sql_call(GOOD_SQL, "b"), sql_call(GOOD_SQL, "c")),
        reply(notes_call()),
    ])
    result = loop(client, db, query_limit=2)
    assert result.queries_used == 2 and result.limit_hit
    results = tool_results(provider.requests[1])
    assert [r["tool_use_id"] for r in results] == ["a", "b", "c"]
    assert [r["is_error"] for r in results] == [False, False, True]
    assert "limit" in results[2]["content"]


def test_loop_results_are_deterministic_sorted_json_marked_untrusted(db, tmp_path):
    client, provider = make_client(tmp_path, [reply(sql_call(GOOD_SQL)), reply(notes_call())])
    loop(client, db)
    text = tool_results(provider.requests[1])[0]["content"]
    body = json.loads(text)
    assert text == json.dumps(body, sort_keys=True)
    assert "untrusted" in body["notice"].lower()
    assert body["rows"] and body["truncated"] is False and body["error"] is None


def test_loop_retries_once_when_no_tool_is_called_then_gives_up(db, tmp_path):
    client, provider = make_client(tmp_path, [
        ModelResponse(text="thinking"), ModelResponse(text="still thinking"),
    ])
    result = loop(client, db)
    assert result.submitted == {} and result.failure
    assert len(provider.requests) == 2
    assert "submit_schema_notes" in provider.requests[1].messages[-1]["content"][0]["text"]


def test_loop_recovers_after_one_malformed_turn(db, tmp_path):
    client, _ = make_client(tmp_path, [ModelResponse(text=""), reply(notes_call())])
    result = loop(client, db)
    assert result.submitted == {"notes": "notes"} and result.failure is None


def test_loop_gives_up_if_forced_submit_is_ignored(db, tmp_path):
    client, _ = make_client(tmp_path, [
        reply(sql_call(GOOD_SQL)), reply(sql_call(GOOD_SQL)), reply(sql_call(GOOD_SQL)),
    ])
    result = loop(client, db, query_limit=1)
    assert result.failure and result.limit_hit


def test_loop_reports_missing_required_fields_back_to_model(db, tmp_path):
    client, provider = make_client(tmp_path, [
        reply(call("submit_schema_notes", "s1")), reply(notes_call()),
    ])
    result = loop(client, db)
    assert result.submitted == {"notes": "notes"}
    (res,) = tool_results(provider.requests[1])
    assert res["is_error"] and "notes" in res["content"]


def test_loop_stops_at_token_limit(db, tmp_path):
    client, provider = make_client(tmp_path, [reply(sql_call(GOOD_SQL), tokens=500)])
    result = loop(client, db, token_limit=500)
    assert result.limit_hit and result.limit_kind == "tokens" and result.submitted == {}
    assert len(provider.requests) == 1


def test_hunt_prompt_names_no_tables_or_recipes():
    assert not re.search(r"GLJRNL|APINVH|GLCOA|DR_AMT|GLJRNH|GLPERD|SYSDICT", HUNT_SYSTEM)
    assert "evidence_sql" in HUNT_SYSTEM and "row keys" in HUNT_SYSTEM


def test_prompts_contain_no_timestamps():
    pattern = re.compile(r"\b(19|20)\d{2}\b|\d{1,2}:\d{2}")
    for text in (HUNT_SYSTEM, EXPLORE_SYSTEM, str(RUN_SQL_TOOL), str(SUBMIT_FINDINGS_TOOL)):
        assert not pattern.search(text)


def test_memo_text_is_marked_untrusted_in_system_prompts():
    assert "untrusted" in HUNT_SYSTEM.lower()
    assert "untrusted" in EXPLORE_SYSTEM.lower()


def base_state(db, **kw):
    state = {
        "dbname": db, "schema_notes": "", "findings": [], "dropped": [], "queue": [],
        "current": None, "attempt": 0, "last_failure": None, "outcomes": [],
        "limits_hit": [],
    }
    state.update(kw)
    return state


def test_explore_stores_notes(db, tmp_path):
    client, _ = make_client(tmp_path, [reply(sql_call(GOOD_SQL)), reply(notes_call("the notes"))])
    out = explore(base_state(db), Deps(client, Limits()))
    assert out["schema_notes"] == "the notes" and out["limits_hit"] == []


def test_explore_records_failure_and_query_limit(db, tmp_path):
    client, _ = make_client(tmp_path, [ModelResponse(text="x"), ModelResponse(text="y")])
    out = explore(base_state(db), Deps(client, Limits()))
    assert out["schema_notes"] == "" and any("explore" in h for h in out["limits_hit"])

    client, _ = make_client(tmp_path, [reply(sql_call(GOOD_SQL)), reply(notes_call())])
    out = explore(base_state(db), Deps(client, Limits(explore_queries=1)))
    assert "explore_queries" in out["limits_hit"] and out["schema_notes"] == "notes"


def test_hunt_assigns_ids_normalises_and_drops_invalid(db, tmp_path):
    items = [
        finding(rows=[("gljrnh", "5")]),
        finding(type="made_up_type"),
        finding(rows=[("GLJRNH", "abc")]),
        finding(type="other", rows=[("apinvh", 2)]),
    ]
    client, provider = make_client(tmp_path, [reply(findings_call(*items))])
    out = hunt(base_state(db, schema_notes="N"), Deps(client, Limits()))
    assert [f.finding_id for f in out["findings"]] == ["F001", "F004"]
    assert out["findings"][0].affected_rows == [RowRef(table="GLJRNH", key=5)]
    assert out["findings"][1].type == "other"
    assert [d.finding_id for d in out["dropped"]] == ["F002", "F003"]
    assert all(d.reason for d in out["dropped"])
    assert "N" in provider.requests[0].messages[0]["content"]


def test_hunt_records_failure_and_keeps_going(db, tmp_path):
    client, _ = make_client(tmp_path, [ModelResponse(text="x"), ModelResponse(text="y")])
    out = hunt(base_state(db), Deps(client, Limits()))
    assert out["findings"] == [] and any("hunt" in h for h in out["limits_hit"])


def f(fid, type, rows, sql=GOOD_SQL):
    return Finding(
        finding_id=fid, type=type, summary=fid,
        affected_rows=[RowRef(table=t, key=k) for t, k in rows], evidence_sql=sql,
    )


def run_confirm(db, tmp_path, findings):
    client, _ = make_client(tmp_path, [])
    return confirm(base_state(db, findings=findings), Deps(client, Limits()))


def test_confirm_drops_finding_whose_evidence_returns_no_rows(db, tmp_path):
    out = run_confirm(db, tmp_path, [f("F001", "other", [("GLJRNH", 1)], EMPTY_SQL)])
    assert out["findings"] == []
    assert "no rows" in out["dropped"][0].reason and out["dropped"][0].finding_id == "F001"


def test_confirm_drops_invalid_evidence_sql_and_logs_it(db, tmp_path):
    out = run_confirm(db, tmp_path, [f("F001", "other", [("GLJRNH", 1)], "SELEC nonsense (")])
    assert out["findings"] == [] and out["dropped"][0].reason


def test_confirm_drops_write_statement_evidence(db, tmp_path):
    out = run_confirm(db, tmp_path, [
        f("F001", "other", [("GLJRNH", 1)], 'DELETE FROM "GLJRNH"'),
    ])
    assert out["findings"] == []
    assert "SELECT" in out["dropped"][0].reason
    with connect_admin(db) as conn:
        assert conn.execute('SELECT count(*) FROM "GLJRNH"').fetchone()[0] > 0


def test_confirm_merges_duplicate_findings(db, tmp_path):
    out = run_confirm(db, tmp_path, [
        f("F001", "duplicate_invoice", [("APINVH", 1), ("APINVH", 2)]),
        f("F002", "duplicate_invoice", [("APINVH", 2), ("APINVH", 3)]),
    ])
    assert len(out["findings"]) == 1
    merged = out["findings"][0]
    assert merged.finding_id == "F001"
    assert merged.affected_rows == [RowRef(table="APINVH", key=k) for k in (1, 2, 3)]


def test_confirm_keeps_same_type_on_different_rows(db, tmp_path):
    out = run_confirm(db, tmp_path, [
        f("F001", "duplicate_invoice", [("APINVH", 1)]),
        f("F002", "duplicate_invoice", [("APINVH", 2)]),
    ])
    assert [x.finding_id for x in out["findings"]] == ["F001", "F002"]


def test_confirm_keeps_different_types_on_shared_rows(db, tmp_path):
    out = run_confirm(db, tmp_path, [
        f("F001", "duplicate_invoice", [("APINVH", 1)]),
        f("F002", "ap_gl_mismatch", [("APINVH", 1)]),
    ])
    assert len(out["findings"]) == 2


def test_confirm_appends_to_existing_dropped(db, tmp_path):
    client, _ = make_client(tmp_path, [])
    from ledger_agent.agent.state import DroppedFinding
    earlier = DroppedFinding(finding_id="F009", summary="s", reason="earlier")
    out = confirm(
        base_state(db, dropped=[earlier], findings=[f("F001", "other", [], EMPTY_SQL)]),
        Deps(client, Limits()),
    )
    assert [d.finding_id for d in out["dropped"]] == ["F009", "F001"]
