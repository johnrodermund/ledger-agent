"""Load a Dataset into a freshly created Postgres database."""

from pathlib import Path

from psycopg import sql

from ledger_agent.db import connect_admin, grant_roles
from ledger_agent.erp.generate import Dataset

HERE = Path(__file__).parent
LOAD_ORDER = ["GLPERD", "GLCOA", "GLJRNH", "GLJRNL", "APVEND", "APINVH", "APPAYM"]


def parse_dictionary(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] == "TBL_NM" or set(cells[0]) <= {"-"}:
            continue
        rows.append({"TBL_NM": cells[0], "COL_NM": cells[1], "DESCR": cells[2]})
    return rows


def _insert(conn, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier(table),
        sql.SQL(", ").join(map(sql.Identifier, cols)),
        sql.SQL(", ").join(sql.Placeholder() * len(cols)))
    with conn.cursor() as cur:
        cur.executemany(query, [[r[c] for c in cols] for r in rows])


def load(dbname: str, dataset: Dataset) -> None:
    """Drop and recreate dbname, then create the schema and load the data."""
    ident = sql.Identifier(dbname)
    with connect_admin("postgres") as admin:
        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(ident))
        admin.execute(sql.SQL("CREATE DATABASE {}").format(ident))
    with connect_admin(dbname) as conn:
        conn.execute((HERE / "schema.sql").read_text(encoding="utf-8"))
        for table in LOAD_ORDER:
            _insert(conn, table, dataset.tables.get(table, []))
        _insert(conn, "SYSDICT", parse_dictionary((HERE / "dictionary.md").read_text(encoding="utf-8")))
        grant_roles(conn)
