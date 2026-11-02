import uuid
from decimal import Decimal

import pytest
from pydantic import ValidationError

from ledger_agent.db import connect_admin
from ledger_agent.erp.faults import INACTIVE_ACCT, MISSING_ACCT, build_scenario
from ledger_agent.erp.generate import N_CLOSED, N_PERIODS, code
from ledger_agent.erp.load import load
from ledger_agent.models import (
    Correction,
    CorrectionLine,
    DecoyType,
    FaultType,
    Finding,
    ScenarioSpec,
    StatusChange,
)
from ledger_agent.verifier.render import Statement, render, render_text
from ledger_agent.verifier.verify import (
    _CHECKS,
    MAX_CORRECTION_ROWS,
    VerifyResult,
    _change_limits,
    _structure,
    _trial_run,
    verify,
)

OPEN_PRD = N_CLOSED + 1
CLOSED_PRD = N_CLOSED
TABLES = ["GLPERD", "GLCOA", "GLJRNH", "GLJRNL", "APVEND", "APINVH", "APPAYM", "SYSDICT"]
RENDER = "ledger_agent.verifier.verify.render"
PIPELINE = "ledger_agent.verifier.verify._CHECKS"


@pytest.fixture(autouse=True)
def first_three_checks_only(monkeypatch):
    """This file tests structure, the trial run and change limits, so its corrections
    need not fix anything. tests/test_verify_rules.py covers the rest."""
    first_three = (_structure, _trial_run, _change_limits)
    assert _CHECKS[:3] == first_three
    monkeypatch.setattr(PIPELINE, first_three)


def line(**overrides):
    return {"acct_cd": "5100", "dr_amt": "0", "cr_amt": "0", "prd_id": OPEN_PRD} | overrides


def debit(acct_cd="5100", amount="10.00", prd_id=OPEN_PRD):
    return CorrectionLine.model_validate(
        line(acct_cd=acct_cd, dr_amt=amount, prd_id=prd_id))


def correction(jrnl_no=None, lines=(), status=()):
    return Correction(corrects_jrnl_no=jrnl_no, lines=list(lines),
                      status_changes=[StatusChange(inv_id=i, new_status="V") for i in status],
                      reasoning="test")


def finding(fault_type=FaultType.UNBALANCED_JOURNAL):
    return Finding(finding_id="X1", type=fault_type, summary="test", affected_rows=[],
                   evidence_sql="SELECT 1")


@pytest.fixture(scope="module")
def scenario():
    spec = ScenarioSpec(id="verify", split="dev", seed=7,
                        faults={FaultType.UNBALANCED_JOURNAL: 1,
                                FaultType.DUPLICATE_INVOICE: 1,
                                FaultType.INVALID_ACCOUNT: 4},
                        decoys=[DecoyType.REVERSING_ENTRY])
    return build_scenario(spec)


@pytest.fixture(scope="module")
def key(scenario):
    return scenario[1]


@pytest.fixture(scope="module")
def db(scenario):
    name = f"verify_{uuid.uuid4().hex[:12]}"
    load(name, scenario[0])
    yield name
    with connect_admin("postgres") as admin:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def primary_journal(fault):
    return next(r.key for r in fault.rows if r.table == "GLJRNH")


def fault_of(key, fault_type):
    return next(f for f in key.faults if f.type == fault_type)


def fault_on_account(key, acct):
    return next(f for f in key.faults
                if any(ln.acct_cd == code(acct) for ln in f.original_lines))


def plain_journal(key):
    return primary_journal(fault_of(key, FaultType.UNBALANCED_JOURNAL))


def duplicate_invoice(key):
    return next(iter(fault_of(key, FaultType.DUPLICATE_INVOICE).expected_status))


def row_counts(db):
    with connect_admin(db) as conn:
        return {t: conn.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in TABLES}


def assert_rejected(db, fix, check, expect=""):
    result = verify(db, finding(), fix)
    assert (result.passed, result.check) == (False, check), result
    assert expect in result.reason
    return result


def assert_passes(db, fix):
    result = verify(db, finding(), fix)
    assert (result.passed, result.check) == (True, "ok"), result


# CorrectionLine parsing

def test_float_amount_parses_exactly():
    assert CorrectionLine.model_validate(line(dr_amt=0.1)).dr_amt == Decimal("0.10")


@pytest.mark.parametrize("raw", [0.1, "0.1", "0.100", 1, Decimal("0.1000")])
def test_amount_quantises_to_two_places_when_exact(raw):
    amount = CorrectionLine.model_validate(line(dr_amt=raw)).dr_amt
    assert amount == Decimal(str(raw)) and amount.as_tuple().exponent == -2


@pytest.mark.parametrize("bad", [
    line(dr_amt="1.234"), line(dr_amt="-1.00"),
    line(dr_amt="1.00", cr_amt="1.00"), line(dr_amt="0", cr_amt="0"),
    line(dr_amt=0.1 + 0.2), line(dr_amt="NaN"), line(dr_amt="Infinity"), line(dr_amt="abc"),
    line(dr_amt="1e30"), line(dr_amt=True), line(dr_amt=None),
])
def test_invalid_line_is_rejected(bad):
    with pytest.raises(ValidationError):
        CorrectionLine.model_validate(bad)


def test_account_code_is_normalised_to_stored_form():
    assert debit(acct_cd="5100  ").acct_cd == "5100    "
    assert debit(acct_cd="5100").acct_cd == "5100    "


def test_status_change_only_allows_void():
    with pytest.raises(ValidationError):
        StatusChange(inv_id=1, new_status="P")


def test_finding_accepts_fault_types_and_other():
    assert finding("other").type == "other"
    assert finding("duplicate_invoice").type is FaultType.DUPLICATE_INVOICE


# Rendering

def test_rendered_sql_is_parameterised():
    fix = correction(7, [debit(amount="1234.56")], [42])
    statements = render(fix, 9000)
    assert "1234.56" not in statements[0].sql
    for stmt in statements:
        assert stmt.sql.count("%s") == len(stmt.params)
        for literal in ("1234.56", "9000", "5100", "'P'", "'GL'", "'V'", "42"):
            assert literal not in stmt.sql
    assert Decimal("1234.56") in statements[1].params


def test_render_writes_header_lines_then_status_changes():
    statements = render(correction(7, [debit(), debit("2000")], [42]), 9000)
    assert [s.sql.split()[:3] for s in statements] == [
        ["INSERT", "INTO", '"GLJRNH"'], ["INSERT", "INTO", '"GLJRNL"'],
        ["INSERT", "INTO", '"GLJRNL"'], ["UPDATE", '"APINVH"', "SET"]]
    assert all(isinstance(s, Statement) for s in statements)
    assert [s.params[1] for s in statements[1:3]] == [1, 2]


def test_render_without_lines_writes_no_journal():
    statements = render(correction(status=[42]), 9000)
    assert len(statements) == 1 and "GLJRN" not in statements[0].sql


def test_status_change_only_touches_stat_cd():
    stmt = render(correction(status=[42]), 9000)[0]
    assert stmt.sql == 'UPDATE "APINVH" SET "STAT_CD" = %s WHERE "INV_ID" = %s'
    assert stmt.params == ("V", 42)


def test_render_text_shows_literal_values():
    text = render_text(correction(7, [debit(amount="1234.56")], [42]), 9000)
    assert "%s" not in text
    for literal in ("9000", "1234.56", "'5100    '", "'V'", "42"):
        assert literal in text


# verify: structure

def test_passing_correction_is_ok(db, key):
    assert_passes(db, correction(plain_journal(key), [debit()]))


def test_verify_returns_a_verify_result(db, key):
    result = verify(db, finding(), correction(plain_journal(key), [debit()]))
    assert isinstance(result, VerifyResult) and result.reason


def test_trimmed_account_code_matches_padded_stored_code(db, key):
    assert_passes(db, correction(plain_journal(key), [debit("5100")]))
    assert_passes(db, correction(plain_journal(key), [debit("5100    ")]))


def test_rejects_unknown_account(db, key):
    assert_rejected(db, correction(plain_journal(key), [debit("9999")]), "structure", "9999")


def test_rejects_inactive_account_not_on_original_journal(db, key):
    fix = correction(plain_journal(key), [debit(INACTIVE_ACCT)])
    assert_rejected(db, fix, "structure", INACTIVE_ACCT)


@pytest.mark.parametrize("acct", [INACTIVE_ACCT, MISSING_ACCT])
def test_allows_reversing_bad_account_already_on_original_journal(db, key, acct):
    jrnl = primary_journal(fault_on_account(key, acct))
    assert_passes(db, correction(jrnl, [CorrectionLine.model_validate(
        line(acct_cd=acct, cr_amt="10.00"))]))


def test_rejects_closed_period(db, key):
    fix = correction(plain_journal(key), [debit(prd_id=CLOSED_PRD)])
    assert_rejected(db, fix, "structure", f"period {CLOSED_PRD}")


def test_rejects_unknown_period(db, key):
    fix = correction(plain_journal(key), [debit(prd_id=N_PERIODS + 1)])
    assert_rejected(db, fix, "structure", f"period {N_PERIODS + 1}")


def test_rejects_lines_in_more_than_one_period(db, key):
    fix = correction(plain_journal(key), [debit(), debit(prd_id=OPEN_PRD + 1)])
    assert_rejected(db, fix, "structure", "one period")


def test_rejects_missing_referenced_journal(db):
    assert_rejected(db, correction(99_999_999, [debit()]), "structure", "99999999")


def test_rejects_correcting_a_correction(db, key):
    reversal = key.decoys[0].rows[0].key
    assert_rejected(db, correction(reversal, [debit()]), "structure", str(reversal))


def test_rejects_no_journal_and_no_status_change(db):
    assert_rejected(db, correction(None, [debit()]), "structure")
    assert_rejected(db, correction(None), "structure")


def test_rejects_lines_without_a_referenced_journal(db, key):
    fix = correction(None, [debit()], [duplicate_invoice(key)])
    assert_rejected(db, fix, "structure", "journal")


def test_rejects_correction_that_changes_nothing(db, key):
    assert_rejected(db, correction(plain_journal(key)), "structure", "nothing")


def test_allows_status_change_without_lines(db, key):
    invoice = duplicate_invoice(key)
    assert_passes(db, correction(None, status=[invoice]))
    assert_passes(db, correction(plain_journal(key), status=[invoice]))


def test_rejects_unknown_invoice(db, key):
    fix = correction(plain_journal(key), [debit()], [99_999_999])
    assert_rejected(db, fix, "structure", "99999999")


def test_rejects_repeated_invoice(db, key):
    invoice = duplicate_invoice(key)
    assert_rejected(db, correction(None, status=[invoice, invoice]), "structure", str(invoice))


def test_rejects_finding_of_type_other(db, key):
    result = verify(db, finding("other"), correction(plain_journal(key), [debit()]))
    assert (result.passed, result.check) == (False, "structure")
    assert "other" in result.reason


# verify: change limits

def test_rejects_too_many_rows(db, key):
    oversized = correction(plain_journal(key), [debit()] * MAX_CORRECTION_ROWS)
    assert_rejected(db, oversized, "change_limits", str(MAX_CORRECTION_ROWS))


def test_allows_rows_at_the_limit(db, key):
    assert_passes(db, correction(plain_journal(key), [debit()] * (MAX_CORRECTION_ROWS - 1)))


@pytest.mark.parametrize("rogue, table", [
    ("INSERT INTO \"SYSDICT\" VALUES ('x', 'y', 'z')", "SYSDICT"),
    ("UPDATE \"GLCOA\" SET \"ACCT_NM\" = 'x'", "GLCOA"),
    ('UPDATE "GLJRNL" SET "DR_AMT" = "DR_AMT" + 1 WHERE "DR_AMT" > 0', "GLJRNL"),
    ("UPDATE \"GLJRNH\" SET \"MEMO\" = 'x'", "GLJRNH"),
    ('UPDATE "APINVH" SET "INV_AMT" = "INV_AMT" + 1', "APINVH"),
    ("UPDATE \"APINVH\" SET \"STAT_CD\" = 'V'", "APINVH"),
])
def test_rejects_writes_beyond_the_rendered_rows(db, key, monkeypatch, rogue, table):
    monkeypatch.setattr(RENDER, lambda fix, n: [*render(fix, n), Statement(rogue, ())])
    fix = correction(None, status=[duplicate_invoice(key)])
    assert_rejected(db, fix, "change_limits", table)


def test_rejects_rendered_rows_that_did_not_land(db, key, monkeypatch):
    monkeypatch.setattr(RENDER, lambda fix, n: render(fix, n)[:-1])
    assert_rejected(db, correction(plain_journal(key), [debit()]), "change_limits", "GLJRNL")


# verify: trial run

def test_database_error_is_a_failed_result(db, key, monkeypatch):
    monkeypatch.setattr(RENDER, lambda fix, n: [Statement("SELECT 1/0", ())])
    assert_rejected(db, correction(plain_journal(key), [debit()]), "trial_run",
                    "division by zero")


def test_database_error_reason_is_repeatable(db, key, monkeypatch):
    monkeypatch.setattr(RENDER, lambda fix, n: [Statement('SELECT * FROM "NOPE"', ())])
    fix = correction(plain_journal(key), [debit()])
    assert verify(db, finding(), fix) == verify(db, finding(), fix)


def test_timeout_is_a_failed_result(db, key, monkeypatch):
    monkeypatch.setattr("ledger_agent.verifier.verify.TRIAL_TIMEOUT_MS", 50)
    monkeypatch.setattr(RENDER, lambda fix, n: [Statement("SELECT pg_sleep(2)", ())])
    assert_rejected(db, correction(plain_journal(key), [debit()]), "trial_run", "50 ms")


def test_lost_connection_is_a_failed_result(db, key, monkeypatch):
    monkeypatch.setattr(RENDER, lambda fix, n: [
        Statement("SELECT pg_terminate_backend(pg_backend_pid())", ())])
    assert_rejected(db, correction(plain_journal(key), [debit()]), "trial_run")


# verify: never commits

def test_always_rolls_back(db, key):
    invoice = duplicate_invoice(key)
    fix = correction(plain_journal(key), [debit()], [invoice])
    before = row_counts(db)
    assert verify(db, finding(), fix).passed
    assert row_counts(db) == before
    with connect_admin(db) as conn:
        stat = conn.execute('SELECT "STAT_CD" FROM "APINVH" WHERE "INV_ID" = %s',
                            (invoice,)).fetchone()[0]
    assert stat == "O"


def test_rolls_back_after_a_failed_check(db, key):
    before = row_counts(db)
    verify(db, finding(), correction(plain_journal(key), [debit()] * MAX_CORRECTION_ROWS))
    assert row_counts(db) == before


def test_next_journal_number_follows_the_highest(db, key, monkeypatch):
    seen = []

    def spying(fix, next_jrnl_no):
        seen.append(next_jrnl_no)
        return render(fix, next_jrnl_no)

    monkeypatch.setattr(RENDER, spying)
    assert verify(db, finding(), correction(plain_journal(key), [debit()])).passed
    with connect_admin(db) as conn:
        top = conn.execute('SELECT max("JRNL_NO") FROM "GLJRNH"').fetchone()[0]
    assert seen == [top + 1]


def test_rendered_journal_has_the_required_header(db, key):
    jrnl = plain_journal(key)
    with connect_admin(db) as conn:
        strt = conn.execute('SELECT "STRT_DT" FROM "GLPERD" WHERE "PRD_ID" = %s',
                            (OPEN_PRD,)).fetchone()[0]
        with conn.transaction(force_rollback=True):
            for stmt in render(correction(jrnl, [debit()]), 900_000):
                conn.execute(stmt.sql, stmt.params)
            header = conn.execute(
                'SELECT "PRD_ID", "POST_DT", "POST_FLG", "REF_JRNL_NO", "SRC_CD" '
                'FROM "GLJRNH" WHERE "JRNL_NO" = 900000').fetchone()
            stored = conn.execute('SELECT "LN_SEQ", "ACCT_CD", "DR_AMT", "CR_AMT" '
                                  'FROM "GLJRNL" WHERE "JRNL_NO" = 900000').fetchall()
    assert header == (OPEN_PRD, strt, "P", jrnl, "GL")
    assert stored == [(1, "5100    ", Decimal("10.00"), Decimal("0.00"))]
