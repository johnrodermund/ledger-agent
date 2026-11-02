"""Database connections and role grants. Host and passwords come from env."""

import os

import psycopg
from psycopg import sql

RO_ROLE = "ledger_ro"
RW_ROLE = "ledger_rw"


class DatabaseUnavailable(Exception):
    """The database cannot be reached or the connection died. Unlike a query error,
    this ends a run: nothing the run reports afterwards could be trusted."""


def _connect(dbname: str, user: str, password: str) -> psycopg.Connection:
    try:
        return psycopg.connect(
            host=os.environ.get("LEDGER_DB_HOST", "localhost"),
            port=os.environ.get("LEDGER_DB_PORT", "5432"),
            dbname=dbname,
            user=user,
            password=password,
            autocommit=True,
        )
    except psycopg.OperationalError as e:
        raise DatabaseUnavailable(
            f"cannot connect to database {dbname!r}: {e.diag.message_primary or str(e)}") from e


_LOST_STATES = ("57P01", "57P02", "57P03")  # admin shutdown, crash shutdown, cannot connect now


def raise_if_unavailable(e: psycopg.Error, conn: psycopg.Connection | None = None) -> None:
    """Raise DatabaseUnavailable if e means the connection is gone. Query-level errors
    (syntax, permissions, timeouts, lock failures) return normally."""
    state = e.sqlstate or ""
    lost = state in _LOST_STATES or state.startswith("08")
    if not lost and isinstance(e, psycopg.OperationalError) and not state:
        lost = True  # no server error: the failure is on the connection itself
    if not lost and conn is not None and conn.broken:
        lost = True
    if lost:
        raise DatabaseUnavailable(f"lost connection to the database: {str(e).strip()}") from e


def connect_admin(dbname: str) -> psycopg.Connection:
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("POSTGRES_PASSWORD is not set; admin access is unavailable here")
    return _connect(dbname, os.environ.get("POSTGRES_USER", "postgres"), password)


def connect_ro(dbname: str) -> psycopg.Connection:
    return _connect(dbname, RO_ROLE, os.environ["LEDGER_RO_PASSWORD"])


def connect_rw(dbname: str) -> psycopg.Connection:
    return _connect(dbname, RW_ROLE, os.environ["LEDGER_RW_PASSWORD"])


def grant_roles(conn: psycopg.Connection) -> None:
    """Grant table privileges in the connected database. Run as admin after
    the tables exist. ledger_ro reads; ledger_rw reads, inserts and updates.
    Neither owns tables, so neither can drop or alter them."""
    conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}, {}").format(
        sql.Identifier(RO_ROLE), sql.Identifier(RW_ROLE)))
    conn.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(
        sql.Identifier(RO_ROLE)))
    conn.execute(sql.SQL(
        "GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO {}").format(
        sql.Identifier(RW_ROLE)))
