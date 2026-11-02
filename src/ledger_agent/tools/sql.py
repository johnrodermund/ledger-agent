"""The only SQL path the LLM gets: one validated SELECT, run read-only.

Output must be deterministic (replay recordings hash it), so queries that
read the clock or random state are rejected, and error text never carries
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
    return None


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
                fetched = cur.fetchmany(max_rows + 1)
    except psycopg.errors.QueryCanceled:
        return _fail(f"query exceeded the {timeout_ms} ms time limit")
    except psycopg.Error as e:
        raise_if_unavailable(e, conn)
        # Primary message only: no connection details, positions or hints.
        return _fail(f"database error: {e.diag.message_primary or type(e).__name__}")
    return QueryResult(
        ok=True,
        columns=columns,
        rows=[[_render(v) for v in row] for row in fetched[:max_rows]],
        truncated=len(fetched) > max_rows,
    )
