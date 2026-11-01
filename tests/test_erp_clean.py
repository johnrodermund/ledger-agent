import uuid
from collections import defaultdict
from decimal import Decimal

import pytest

from ledger_agent.db import connect_admin
from ledger_agent.erp.generate import generate_clean
from ledger_agent.erp.load import load


@pytest.fixture(scope="module")
def loaded_db():
    name = f"erp_{uuid.uuid4().hex[:12]}"
    load(name, generate_clean(7))
    try:
        yield name
    finally:
        with connect_admin("postgres") as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def fetch_one(db, query):
    with connect_admin(db) as c:
        return c.execute(query).fetchone()


def count(db, table, where):
    return fetch_one(db, f"SELECT count(*) FROM {table} WHERE {where}")[0]


def test_generation_is_deterministic():
    assert generate_clean(7) == generate_clean(7)


def test_different_seeds_differ():
    assert generate_clean(7) != generate_clean(8)


def test_every_clean_journal_balances():
    t = generate_clean(7).tables
    dr, cr = defaultdict(Decimal), defaultdict(Decimal)
    for ln in t["GLJRNL"]:
        dr[ln["JRNL_NO"]] += ln["DR_AMT"]
        cr[ln["JRNL_NO"]] += ln["CR_AMT"]
    assert {h["JRNL_NO"] for h in t["GLJRNH"]} == set(dr)
    assert all(dr[j] == cr[j] for j in dr)


def test_every_invoice_ties_to_its_journal_ap_line():
    t = generate_clean(7).tables
    ap_credit = defaultdict(Decimal)
    for ln in t["GLJRNL"]:
        if ln["ACCT_CD"].strip() == "2000":
            ap_credit[ln["JRNL_NO"]] += ln["CR_AMT"]
    assert t["APINVH"]
    for inv in t["APINVH"]:
        assert ap_credit[inv["JRNL_NO"]] == inv["INV_AMT"]


def test_money_is_two_place_decimal():
    t = generate_clean(7).tables
    for ln in t["GLJRNL"]:
        for k in ("DR_AMT", "CR_AMT"):
            assert isinstance(ln[k], Decimal) and ln[k].as_tuple().exponent == -2


def test_periods_and_posting_dates():
    t = generate_clean(7).tables
    periods = {p["PRD_ID"]: p for p in t["GLPERD"]}
    assert len(periods) == 12
    assert [periods[i]["STAT_CD"] for i in range(1, 13)] == ["C"] * 10 + ["O"] * 2
    for h in t["GLJRNH"]:
        p = periods[h["PRD_ID"]]
        assert p["STRT_DT"] <= h["POST_DT"] <= p["END_DT"]
        assert h["POST_FLG"] == "P"
        if p["CLOSE_DT"] is not None:
            assert h["POST_DT"] < p["CLOSE_DT"]


def test_line_count_is_between_2000_and_5000():
    assert 2000 <= len(generate_clean(7).tables["GLJRNL"]) <= 5000


def test_schema_declares_no_foreign_keys(loaded_db):
    assert count(loaded_db, "information_schema.table_constraints",
                 "constraint_type = 'FOREIGN KEY'") == 0


def test_codes_are_space_padded(loaded_db):
    assert len(fetch_one(loaded_db, 'SELECT "ACCT_CD" FROM "GLCOA" LIMIT 1')[0]) == 8


def test_load_is_repeatable_and_grants_read(loaded_db):
    load(loaded_db, generate_clean(8))
    assert fetch_one(loaded_db, 'SELECT count(*) FROM "APINVH"')[0] > 0
    from ledger_agent.db import connect_ro
    with connect_ro(loaded_db) as c:
        assert c.execute('SELECT count(*) FROM "SYSDICT"').fetchone()[0] > 0


def test_sysdict_omits_ref_jrnl_no_and_close_dt_and_misdescribes_src_ref(loaded_db):
    assert count(loaded_db, '"SYSDICT"', "\"COL_NM\" IN ('REF_JRNL_NO','CLOSE_DT')") == 0
    descr = fetch_one(loaded_db, "SELECT \"DESCR\" FROM \"SYSDICT\" WHERE \"COL_NM\"='SRC_REF'")[0]
    assert "user" in descr.lower()
