"""The only SQL path the LLM gets: one validated SELECT, run read-only.

Output must be deterministic (replay recordings hash it), so queries that read the
clock, random state, planner statistics, row versions or server identity are rejected,
rows without a top-level ORDER BY are sorted here, and error text never carries
connection-specific detail.
"""

from dataclasses import dataclass, field

import psycopg
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from ledger_agent.db import connect_ro, raise_if_unavailable

NON_DETERMINISTIC = {
    "now", "random", "current_date", "current_time", "current_timestamp",
    "localtime", "localtimestamp", "clock_timestamp", "timeofday",
    "statement_timestamp", "transaction_timestamp", "gen_random_uuid",
}
TIME_LITERALS = {"now", "today", "tomorrow", "yesterday"}
NON_DET_MESSAGE = "non-deterministic functions are not allowed (results must be repeatable)"
NOT_REPEATABLE_MESSAGE = (
    "statistics, row versions and server state are not allowed (results must be repeatable)")
SAMPLE_MESSAGE = "TABLESAMPLE needs REPEATABLE (results must be repeatable)"
NOT_REPEATABLE_FUNCTIONS = {
    "txid_current", "pg_current_xact_id", "pg_backend_pid", "pg_postmaster_start_time",
    "inet_server_addr", "inet_server_port", "pg_relation_size", "pg_total_relation_size",
    "pg_database_size", "pg_table_size", "version",
}
NOT_REPEATABLE_RELATIONS = {"pg_locks", "pg_class", "pg_database"}
NOT_REPEATABLE_RELATION_PREFIX = "pg_stat"  # also covers pg_statio_*, pg_stats, pg_statistic
SYSTEM_COLUMNS = {"xmin", "xmax", "cmin", "cmax", "ctid", "oid", "tableoid"}
FORBIDDEN_FUNCTIONS = {
    "pg_read_file", "pg_ls_dir", "lo_import", "lo_export", "set_config", "dblink",
}
WRITE_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop,
    exp.Alter, exp.Command, exp.Into, exp.Copy, exp.Set, exp.TruncateTable,
)
# sqlglot maps some SQL functions to its own nodes; normalise those names.
NODE_ALIASES = {
    "currentdate": "current_date", "currenttime": "current_time",
    "currenttimestamp": "current_timestamp", "rand": "random",
    "currentdatetime": "current_timestamp", "uuid": "gen_random_uuid",
}


@dataclass
class QueryResult:
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    truncated: bool = False
    error: str | None = None


def _fail(message: str) -> QueryResult:
    return QueryResult(ok=False, error=message)


def _function_names(node: exp.Expression) -> set[str]:
    names = set()
    if isinstance(node, exp.Anonymous):
        names.add(node.name.lower())
    if isinstance(node, exp.Func):
        names.add(node.sql_name().lower())
    names.add(type(node).__name__.lower())
    # LOCALTIME and LOCALTIMESTAMP parse as bare unquoted column references.
    if isinstance(node, exp.Column) and not node.table and not node.this.quoted:
        names.add(node.name.lower())
    return {NODE_ALIASES.get(n, n) for n in names}


def _not_repeatable(node: exp.Expression) -> str | None:
    """Why node gives different results on an identical database, or None."""
    if isinstance(node, exp.TableSample) and node.args.get("seed") is None:
        return SAMPLE_MESSAGE
    if isinstance(node, exp.Table):
        name = node.name.lower()
        if name in NOT_REPEATABLE_RELATIONS or name.startswith(NOT_REPEATABLE_RELATION_PREFIX):
            return NOT_REPEATABLE_MESSAGE
    if isinstance(node, exp.Column) and node.name.lower() in SYSTEM_COLUMNS:
        return NOT_REPEATABLE_MESSAGE
    # Only a call counts: a plain column may be called version.
    if isinstance(node, exp.Func) and _function_names(node) & NOT_REPEATABLE_FUNCTIONS:
        return NOT_REPEATABLE_MESSAGE
    return None


def validate(sql: str) -> str | None:
    """Return an error message, or None if sql is one safe, deterministic SELECT."""
    try:
        statements = sqlglot.parse(sql, read="postgres")
    except SqlglotError:
        return "SQL could not be parsed"
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        return "exactly one SELECT statement is required"
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        return "only SELECT statements are allowed"
    for node in tree.walk():
        if isinstance(node, WRITE_NODES):
            return "only SELECT statements are allowed"
        if isinstance(node, exp.Lock):
            return "locking clauses such as FOR UPDATE are not allowed"
        if (isinstance(node, exp.Cast) and isinstance(node.this, exp.Literal)
                and node.this.is_string
                and node.this.name.strip().lower() in TIME_LITERALS):
            return NON_DET_MESSAGE
        names = _function_names(node)
        if names & NON_DETERMINISTIC:
            return NON_DET_MESSAGE
        if names & FORBIDDEN_FUNCTIONS:
            return "file and session functions are not allowed"
        reason = _not_repeatable(node)
        if reason:
            return reason
    return None


def _is_ordered(sql: str) -> bool:
    """Whether a validated statement has an ORDER BY at the top level."""
    return sqlglot.parse_one(sql, read="postgres").args.get("order") is not None


def _render(value) -> str:
    return "NULL" if value is None else str(value)


def run_query(dbname: str, sql: str, max_rows: int = 200, timeout_ms: int = 5000) -> QueryResult:
    problem = validate(sql)
    if problem:
        return _fail(problem)
    conn = None
    try:
        with connect_ro(dbname) as conn:
            with conn.transaction():
                conn.execute("SET TRANSACTION READ ONLY")
                conn.execute("SET LOCAL TimeZone = 'UTC'")
                conn.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
                conn.execute("SET LOCAL max_parallel_workers_per_gather = 0")
                cur = conn.execute(sql)
                columns = [d.name for d in cur.description]
                fetched = cur.fetchall()
    except psycopg.errors.QueryCanceled:
        return _fail(f"query exceeded the {timeout_ms} ms time limit")
    except psycopg.Error as e:
        raise_if_unavailable(e, conn)
        # Primary message only: no connection details, positions or hints.
        return _fail(f"database error: {e.diag.message_primary or type(e).__name__}")
    rows = [[_render(v) for v in row] for row in fetched]
    if not _is_ordered(sql):
        rows.sort()  # before the cap, so the same rows survive whatever plan was chosen
    return QueryResult(ok=True, columns=columns, rows=rows[:max_rows],
                       truncated=len(rows) > max_rows)
