import psycopg
import pytest

from ledger_agent.db import connect_admin, connect_ro, connect_rw


def test_ro_role_cannot_write(scratch_db):
    with connect_ro(scratch_db) as c, pytest.raises(psycopg.errors.InsufficientPrivilege):
        c.execute("INSERT INTO probe VALUES (1)")


def test_rw_role_can_insert_and_update(scratch_db):
    with connect_rw(scratch_db) as c:
        c.execute("INSERT INTO probe VALUES (1)")
        c.execute("UPDATE probe SET id = 2")
        assert c.execute("SELECT id FROM probe").fetchall() == [(2,)]
    with connect_ro(scratch_db) as c:
        assert c.execute("SELECT count(*) FROM probe").fetchone() == (1,)


def test_rw_role_cannot_drop_table(scratch_db):
    with connect_rw(scratch_db) as c, pytest.raises(psycopg.errors.InsufficientPrivilege):
        c.execute("DROP TABLE probe")
    with connect_admin(scratch_db) as c:
        assert c.execute("SELECT to_regclass('probe')").fetchone()[0] is not None
