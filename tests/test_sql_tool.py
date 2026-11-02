import uuid

import pytest

from ledger_agent.db import connect_admin
from ledger_agent.erp.generate import generate_clean
from ledger_agent.erp.load import load
from ledger_agent.tools.sql import run_query


@pytest.fixture(scope="module")
def db():
    name = f"sqltool_{uuid.uuid4().hex[:12]}"
    load(name, generate_clean(7))
    try:
        yield name
    finally:
        with connect_admin("postgres") as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.mark.parametrize("sql", [
    "DELETE FROM \"GLJRNL\"",
    "SELECT 1; DROP TABLE \"GLJRNL\"",
    "WITH x AS (DELETE FROM \"GLJRNL\" RETURNING *) SELECT * FROM x",
    "SELECT * INTO copy FROM \"GLJRNL\"",
    "SELECT * FROM \"GLJRNL\" FOR UPDATE",
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT set_config('statement_timeout', '0', false)",
    "",
    "SELECT pg_catalog.pg_read_file('x')",
    "SELECT \"pg_read_file\"('x')",
    "SELECT \"set_config\"('a','b',false)",
    "SELECT 1 FOR SHARE",
    "SELECT dblink('x')",
    "SELECT lo_export(1,'x')",
])
def test_rejects_non_select(db, sql):
    r = run_query(db, sql)
    assert r.ok is False and r.error


@pytest.mark.parametrize("sql", [
    "SELECT now()",
    "SELECT random()",
    "SELECT CURRENT_DATE",
    "SELECT CURRENT_TIME",
    "SELECT CURRENT_TIMESTAMP",
    "SELECT LOCALTIME",
    "SELECT LOCALTIMESTAMP",
    "SELECT clock_timestamp()",
    "SELECT timeofday()",
    "SELECT statement_timestamp()",
    "SELECT transaction_timestamp()",
    "SELECT gen_random_uuid()",
    "SELECT 1 WHERE EXISTS (SELECT now())",
    "SELECT NOW()",
    "SELECT pg_catalog.now()",
    'SELECT "now"()',
    'SELECT "random"()',
    "SELECT 1 UNION SELECT random()",
    "WITH a AS (SELECT now()) SELECT * FROM a",
    "SELECT 'now'::timestamp",
    "SELECT ' Today '::date",
    "SELECT CAST('today' AS date)",
    "SELECT 'tomorrow'::timestamptz",
])
def test_rejects_nondeterministic_functions(db, sql):
    # replay depends on identical tool output
    assert "non-deterministic" in run_query(db, sql).error


@pytest.mark.parametrize("sql", [
    'SELECT * FROM "GLJRNL" TABLESAMPLE BERNOULLI (10)',
    'SELECT * FROM "GLJRNL" TABLESAMPLE SYSTEM (10)',
    "SELECT * FROM pg_stat_user_tables",
    "SELECT * FROM pg_stat_activity",
    "SELECT * FROM pg_catalog.pg_statio_user_tables",
    "SELECT * FROM pg_stats",
    "SELECT * FROM pg_locks",
    "SELECT reltuples FROM pg_class",
    "SELECT * FROM pg_catalog.pg_database",
    "SELECT * FROM \"pg_class\"",
    'SELECT xmin FROM "GLJRNL"',
    'SELECT xmax FROM "GLJRNL"',
    'SELECT cmin FROM "GLJRNL"',
    'SELECT cmax FROM "GLJRNL"',
    'SELECT ctid FROM "GLJRNL"',
    'SELECT l.ctid FROM "GLJRNL" l',
    'SELECT "xmin" FROM "GLJRNL"',
    'SELECT oid FROM pg_type',
    'SELECT tableoid FROM "GLJRNL"',
    'SELECT 1 FROM "GLJRNL" ORDER BY ctid',
    "SELECT txid_current()",
    "SELECT pg_current_xact_id()",
    "SELECT pg_backend_pid()",
    "SELECT pg_postmaster_start_time()",
    "SELECT inet_server_addr()",
    "SELECT inet_server_port()",
    "SELECT pg_relation_size('\"GLJRNL\"')",
    "SELECT pg_total_relation_size('\"GLJRNL\"')",
    "SELECT pg_database_size('x')",
    "SELECT pg_table_size('\"GLJRNL\"')",
    "SELECT version()",
    "SELECT VERSION()",
    'SELECT "version"()',
    "SELECT pg_catalog.version()",
])
def test_rejects_anything_that_varies_between_runs(db, sql):
    assert "repeatable" in run_query(db, sql).error


@pytest.mark.parametrize("sql", [
    'SELECT "JRNL_NO" FROM "GLJRNL" TABLESAMPLE BERNOULLI (10) REPEATABLE (7)',
    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
    "SELECT column_name, data_type FROM information_schema.columns",
    "SELECT pg_sleep(0)",
    "SELECT 1 AS version",
])
def test_repeatable_queries_stay_usable(db, sql):
    assert run_query(db, sql).ok


def test_a_column_named_version_is_not_a_function_call():
    from ledger_agent.tools.sql import validate

    assert validate('SELECT version FROM "SOME_TABLE"') is None


def test_syntax_error_is_returned_not_raised(db):
    assert run_query(db, "SELEC 1").ok is False


def test_database_error_is_returned_not_raised(db):
    r = run_query(db, 'SELECT * FROM "NOSUCH"')
    assert r.ok is False and r.error


def test_error_text_is_deterministic(db):
    assert run_query(db, 'SELECT * FROM "NOSUCH"').error == run_query(db, 'SELECT * FROM "NOSUCH"').error


def test_caps_rows(db):
    r = run_query(db, 'SELECT * FROM "GLJRNL"', max_rows=5)
    assert len(r.rows) == 5 and r.truncated


def test_not_truncated_when_within_cap(db):
    r = run_query(db, "SELECT 1 UNION SELECT 2", max_rows=2)
    assert r.ok and len(r.rows) == 2 and not r.truncated


def test_times_out(db):
    r = run_query(db, "SELECT pg_sleep(1)", timeout_ms=50)
    assert r.ok is False and r.error == "query exceeded the 50 ms time limit"


def test_plain_string_literals_are_allowed(db):
    assert run_query(db, "SELECT 'now'").rows == [["now"]]


def test_timestamps_render_in_utc(db):
    assert run_query(db, "SELECT '2024-01-01 00:00:00+05'::timestamptz").rows == [["2023-12-31 19:00:00+00:00"]]


def test_decimals_render_exactly(db):
    assert run_query(db, "SELECT 1.10::numeric(14,2)").rows == [["1.10"]]


def test_columns_and_null(db):
    r = run_query(db, "SELECT 1 AS a, NULL AS b")
    assert r.ok and r.columns == ["a", "b"] and r.rows == [["1", "NULL"]]


def test_char_padding_is_preserved(db):
    assert run_query(db, "SELECT 'ab'::char(4)").rows == [["ab  "]]


def test_connection_stays_read_only(db):
    # a trailing comment must not smuggle a write past the parser
    r = run_query(db, 'SELECT 1 -- ; DELETE FROM "GLJRNL"')
    assert r.ok and r.rows == [["1"]]


def test_a_leading_semicolon_does_not_crash_the_order_check(db):
    r = run_query(db, '; SELECT "JRNL_NO" FROM "GLJRNH" ORDER BY 1 DESC LIMIT 3')
    assert r.ok and r.rows == sorted(r.rows, reverse=True)
