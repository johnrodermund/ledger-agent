"""Database connections and role grants. Host and passwords come from env."""

import os

import psycopg
from psycopg import sql

RO_ROLE = "ledger_ro"
RW_ROLE = "ledger_rw"


def _connect(dbname: str, user: str, password: str) -> psycopg.Connection:
    return psycopg.connect(
        host=os.environ.get("LEDGER_DB_HOST", "localhost"),
        port=os.environ.get("LEDGER_DB_PORT", "5432"),
        dbname=dbname,
        user=user,
        password=password,
        autocommit=True,
    )


def connect_admin(dbname: str) -> psycopg.Connection:
    return _connect(
        dbname,
        os.environ.get("POSTGRES_USER", "postgres"),
        os.environ["POSTGRES_PASSWORD"],
    )


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
