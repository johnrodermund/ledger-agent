"""Tool output must not depend on planner statistics, or replays stop matching."""

import uuid

import pytest

from ledger_agent.agent.loop import format_query_result
from ledger_agent.db import connect_admin
from ledger_agent.erp.faults import build_scenario
from ledger_agent.erp.load import load
from ledger_agent.models import DecoyType, FaultType, ScenarioSpec
from ledger_agent.tools.sql import run_query

UNORDERED = [
    'SELECT "JRNL_NO", sum("DR_AMT") - sum("CR_AMT") FROM "GLJRNL" GROUP BY "JRNL_NO"',
    'SELECT "ACCT_CD", count(*) FROM "GLJRNL" GROUP BY "ACCT_CD"',
    'SELECT "VEND_CD", "INV_NO", count(*) FROM "APINVH" GROUP BY "VEND_CD", "INV_NO"',
    'SELECT DISTINCT "ACCT_CD" FROM "GLJRNL"',
    'SELECT DISTINCT "VEND_CD", "STAT_CD" FROM "APINVH"',
    'SELECT h."JRNL_NO", h."MEMO", l."ACCT_CD", l."DR_AMT" FROM "GLJRNH" h '
    'JOIN "GLJRNL" l ON l."JRNL_NO" = h."JRNL_NO"',
    'SELECT i."INV_ID", i."INV_AMT", sum(l."CR_AMT") FROM "APINVH" i '
    'JOIN "GLJRNL" l ON l."JRNL_NO" = i."JRNL_NO" GROUP BY i."INV_ID", i."INV_AMT"',
    'SELECT p."PAY_ID", i."VEND_CD" FROM "APPAYM" p LEFT JOIN "APINVH" i ON i."INV_ID" = p."INV_ID"',
    'SELECT "ACCT_CD" FROM "GLCOA" UNION SELECT "ACCT_CD" FROM "GLJRNL"',
]


@pytest.fixture(scope="module")
def db():
    spec = ScenarioSpec(id="determinism", split="dev", seed=101,
                        faults={t: 1 for t in FaultType}, decoys=list(DecoyType))
    name = f"determinism_{uuid.uuid4().hex[:12]}"
    load(name, build_scenario(spec)[0])
    try:
        yield name
    finally:
        with connect_admin("postgres") as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def outputs(db):
    return [format_query_result(run_query(db, sql)) for sql in UNORDERED]


def test_tool_output_is_the_same_after_analyze(db):
    before = outputs(db)
    with connect_admin(db) as conn:
        conn.execute("ANALYZE")
    assert outputs(db) == before


def test_load_leaves_planner_statistics_in_place(db):
    with connect_admin(db) as conn:
        analyzed = conn.execute(
            "SELECT count(*) FROM pg_stat_user_tables WHERE analyze_count + autoanalyze_count > 0"
        ).fetchone()[0]
    assert analyzed == 8


def test_unordered_rows_are_sorted_before_the_cap(db):
    full = run_query(db, 'SELECT DISTINCT "ACCT_CD" FROM "GLJRNL"', max_rows=1000)
    capped = run_query(db, 'SELECT DISTINCT "ACCT_CD" FROM "GLJRNL"', max_rows=3)
    assert full.rows == sorted(full.rows)
    assert capped.rows == full.rows[:3] and capped.truncated


def test_an_explicit_order_by_is_kept(db):
    result = run_query(db, 'SELECT "JRNL_NO" FROM "GLJRNH" ORDER BY "JRNL_NO" DESC', max_rows=3)
    numbers = [int(row[0]) for row in result.rows]
    assert numbers == sorted(numbers, reverse=True) and result.truncated


def test_a_union_order_by_is_kept(db):
    result = run_query(db, "SELECT 1 AS a UNION SELECT 2 ORDER BY a DESC")
    assert result.rows == [["2"], ["1"]]


def test_an_inner_order_by_does_not_stop_the_sort(db):
    result = run_query(db, "SELECT a FROM (SELECT 1 AS a UNION SELECT 2 ORDER BY 1 DESC) s")
    assert result.rows == [["1"], ["2"]]
