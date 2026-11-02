import ast
import uuid
from decimal import Decimal
from pathlib import Path

import pytest

import ledger_agent.verifier
from ledger_agent.db import connect_admin, connect_rw
from ledger_agent.erp.faults import INACTIVE_ACCT, MISSING_ACCT, build_scenario
from ledger_agent.erp.generate import AP_CONTROL, N_CLOSED, Dataset, code
from ledger_agent.erp.load import load
from ledger_agent.models import (
    AnswerKey,
    Correction,
    CorrectionLine,
    DecoyType,
    FaultType,
    Finding,
    PlantedFault,
    ScenarioSpec,
    StatusChange,
)
from ledger_agent.verifier.invariants import (
    AP_ACCOUNT,
    PAYMENT_TOLERANCE,
    Violation,
    global_violations,
)
from ledger_agent.verifier.rules import check_resolved
from ledger_agent.verifier.verify import verify

OPEN_PRD = N_CLOSED + 1
CLOSED_PRD = N_CLOSED


def spec(faults, decoys=(), seed=7):
    return ScenarioSpec(id="rules", split="dev", seed=seed, faults=faults, decoys=list(decoys))


@pytest.fixture(scope="module")
def loaded():
    """Load scenarios on demand and drop them all when the module ends."""
    cache: dict[str, tuple[str, Dataset, AnswerKey]] = {}

    def get(scenario: ScenarioSpec) -> tuple[str, Dataset, AnswerKey]:
        cache_key = scenario.model_dump_json()
        if cache_key not in cache:
            name = f"rules_{uuid.uuid4().hex[:12]}"
            dataset, answers = build_scenario(scenario)
            load(name, dataset)
            cache[cache_key] = (name, dataset, answers)
        return cache[cache_key]

    yield get
    with connect_admin("postgres") as admin:
        for name, _, _ in cache.values():
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture(scope="module")
def db_with_fault(loaded):
    def get(fault_type: FaultType) -> tuple[str, Dataset, PlantedFault]:
        name, dataset, answers = loaded(spec({fault_type: 1}))
        return name, dataset, answers.faults[0]

    return get


@pytest.fixture(scope="module")
def db_with_two_faults(loaded):
    name, _, answers = loaded(spec({FaultType.AP_GL_MISMATCH: 1,
                                    FaultType.INVALID_ACCOUNT: 1}))
    return name, answers.faults


def debit(acct, amount, prd_id=OPEN_PRD):
    return CorrectionLine(acct_cd=acct, dr_amt=amount, cr_amt=0, prd_id=prd_id)


def credit(acct, amount, prd_id=OPEN_PRD):
    return CorrectionLine(acct_cd=acct, dr_amt=0, cr_amt=amount, prd_id=prd_id)


def correction(jrnl_no, lines=(), status=()):
    return Correction(corrects_jrnl_no=jrnl_no, lines=list(lines),
                      status_changes=[StatusChange(inv_id=i, new_status="V") for i in status],
                      reasoning="test")


def finding_of(fault_type, rows=()):
    return Finding(finding_id="R1", type=fault_type, summary="test",
                   affected_rows=list(rows), evidence_sql="SELECT 1")


def journal(fault):
    return next(r.key for r in fault.rows if r.table == "GLJRNH")


def true_up(fault, prd_id=OPEN_PRD):
    """The lines that take the journal from where it stands to expected_net."""
    lines = []
    for acct, want in sorted(fault.expected_net.items()):
        have = sum((k.dr_amt - k.cr_amt for k in fault.original_lines if k.acct_cd == acct),
                   Decimal("0.00"))
        if want > have:
            lines.append(debit(acct, want - have, prd_id))
        elif want < have:
            lines.append(credit(acct, have - want, prd_id))
    return lines


def correct_fix(fault):
    return correction(journal(fault), true_up(fault), fault.expected_status)


def run(db, fault, fix):
    return verify(db, finding_of(fault.type, fault.rows), fix)


def assert_rejected(db, fault, fix, check, expect=""):
    result = run(db, fault, fix)
    assert (result.passed, result.check) == (False, check), result
    assert expect in result.reason
    return result


def faulted_kinds(db):
    with connect_rw(db) as conn:
        return {v.kind for v in global_violations(conn)}


def find_unbalanced(loaded, on_invoice):
    """A scenario whose unbalanced journal is, or is not, an invoice journal."""
    for seed in range(1, 21):
        name, dataset, answers = loaded(spec({FaultType.UNBALANCED_JOURNAL: 1}, seed=seed))
        fault = answers.faults[0]
        if any(r.table == "APINVH" for r in fault.rows) == on_invoice:
            return name, dataset, fault
    pytest.fail("no seed planted that kind of unbalanced journal")


def ap_plug(fault):
    """The true correction's amount and side, on the AP control account."""
    wanted = true_up(fault)[0]
    return (debit if wanted.dr_amt else credit)(AP_CONTROL, wanted.dr_amt or wanted.cr_amt)


# Resolution

@pytest.mark.parametrize("fault_type", list(FaultType))
def test_correct_fix_passes(fault_type, db_with_fault):
    db, _, fault = db_with_fault(fault_type)
    result = run(db, fault, correct_fix(fault))
    assert (result.passed, result.check) == (True, "ok"), result


@pytest.mark.parametrize("fault_type", list(FaultType))
def test_empty_fix_fails_resolution(fault_type, db_with_fault):
    """Lines that net to nothing, posted against the right journal."""
    db, _, fault = db_with_fault(fault_type)
    noop = [debit("5100", "10.00"), credit("5100", "10.00")]
    assert_rejected(db, fault, correction(journal(fault), noop), "resolution")


def test_correcting_a_journal_that_was_never_faulted_fails_resolution(db_with_fault):
    db, dataset, fault = db_with_fault(FaultType.UNBALANCED_JOURNAL)
    other = next(h["JRNL_NO"] for h in dataset.tables["GLJRNH"]
                 if h["SRC_CD"].strip() == "GL" and h["JRNL_NO"] != journal(fault))
    fix = correction(other, [debit("5100", "10.00"), credit("5100", "10.00")])
    assert_rejected(db, fault, fix, "resolution", "clears no")


def test_no_op_on_a_healthy_invoice_journal_fails_resolution(db_with_fault):
    db, dataset, fault = db_with_fault(FaultType.AP_GL_MISMATCH)
    cited = {r.key for r in fault.rows if r.table == "GLJRNH"}
    healthy = next(i["JRNL_NO"] for i in dataset.tables["APINVH"] if i["JRNL_NO"] not in cited)
    fix = correction(healthy, [debit("5100", "10.00"), credit("5100", "10.00")])
    assert_rejected(db, fault, fix, "resolution", "clears no")


def test_void_without_reversal_fails_resolution(db_with_fault):
    db, _, fault = db_with_fault(FaultType.DUPLICATE_INVOICE)
    assert_rejected(db, fault, correction(None, status=fault.expected_status), "resolution")


def test_reversal_without_void_fails_resolution(db_with_fault):
    db, _, fault = db_with_fault(FaultType.DUPLICATE_INVOICE)
    assert_rejected(db, fault, correction(journal(fault), true_up(fault)), "resolution", "void")


def test_voiding_both_copies_fails_resolution(db_with_fault):
    db, _, fault = db_with_fault(FaultType.DUPLICATE_INVOICE)
    both = [r.key for r in fault.rows if r.table == "APINVH"]
    assert_rejected(db, fault, correction(journal(fault), true_up(fault), both), "resolution")


def test_ap_mismatch_needs_the_group_to_balance(db_with_fault):
    db, _, fault = db_with_fault(FaultType.AP_GL_MISMATCH)
    ap_only = [ln for ln in true_up(fault) if ln.acct_cd == AP_CONTROL.ljust(8)]
    assert_rejected(db, fault, correction(journal(fault), ap_only), "resolution", "balance")


def test_invalid_account_cleared_without_balancing_fails_resolution(db_with_fault):
    db, _, fault = db_with_fault(FaultType.INVALID_ACCOUNT)
    bad = next(a for a, v in fault.expected_net.items() if v == 0)
    lines = [ln for ln in true_up(fault) if ln.acct_cd == bad]
    assert_rejected(db, fault, correction(journal(fault), lines), "resolution", "balance")


# The rules read stored rows

def test_rules_judge_stored_rows_not_the_correction(db_with_fault):
    db, _, fault = db_with_fault(FaultType.UNBALANCED_JOURNAL)
    with connect_rw(db) as conn:
        # Nothing was written, so a correction that would fix it does not count.
        reason = check_resolved(conn, finding_of(fault.type), correct_fix(fault))
    assert reason is not None and "balance" in reason


def test_closed_rule_rejects_stored_lines_in_a_closed_period(db_with_fault):
    db, _, fault = db_with_fault(FaultType.CLOSED_PERIOD_POSTING)
    jrnl, fix = journal(fault), correct_fix(fault)
    with connect_admin(db) as conn, conn.transaction(force_rollback=True):
        conn.execute('INSERT INTO "GLJRNH" ("JRNL_NO", "PRD_ID", "POST_DT", "POST_FLG", '
                     '"REF_JRNL_NO", "SRC_CD") VALUES (900000, %s, 20240101, %s, %s, %s)',
                     (CLOSED_PRD, "P", jrnl, "GL"))
        for seq, ln in enumerate(fix.lines, start=1):
            conn.execute('INSERT INTO "GLJRNL" VALUES (900000, %s, %s, %s, %s)',
                         (seq, ln.acct_cd, ln.dr_amt, ln.cr_amt))
        reason = check_resolved(conn, finding_of(fault.type), fix)
    assert reason is not None and str(CLOSED_PRD) in reason


def test_rules_refuse_a_finding_of_type_other(db_with_fault):
    db, _, fault = db_with_fault(FaultType.UNBALANCED_JOURNAL)
    with connect_rw(db) as conn:
        assert check_resolved(conn, finding_of("other"), correct_fix(fault))


# Collateral

def test_fix_introducing_new_violation_fails_collateral(db_with_fault):
    """Voiding the paid original resolves the duplicate but strands its payment."""
    db, dataset, fault = db_with_fault(FaultType.DUPLICATE_INVOICE)
    copy = next(iter(fault.expected_status))
    original = next(r.key for r in fault.rows if r.table == "APINVH" and r.key != copy)
    invoice = next(i for i in dataset.tables["APINVH"] if i["INV_ID"] == original)
    fix = correction(invoice["JRNL_NO"], true_up(fault), [original])
    result = assert_rejected(db, fault, fix, "collateral", "payment_exceeds_invoice")
    assert str(original) in result.reason


def test_plugging_an_invoice_journal_to_ap_fails_collateral(loaded):
    db, _, fault = find_unbalanced(loaded, on_invoice=True)
    fix = correction(journal(fault), [ap_plug(fault)])
    assert_rejected(db, fault, fix, "collateral", "ap_gl_mismatch")


def test_preexisting_other_faults_do_not_fail_collateral(db_with_two_faults):
    db, faults = db_with_two_faults
    for fault in faults:
        result = run(db, fault, correct_fix(fault))
        assert (result.passed, result.check) == (True, "ok"), result


# Global checks

@pytest.mark.parametrize("seed", [7, 8, 9])
def test_clean_scenario_with_all_decoys_has_no_violations(loaded, seed):
    db, _, _ = loaded(spec({}, list(DecoyType) * 2, seed=seed))
    with connect_rw(db) as conn:
        assert global_violations(conn) == set()


@pytest.mark.parametrize("fault_type", list(FaultType))
def test_a_planted_fault_is_a_violation_of_its_own_kind_only(fault_type, db_with_fault):
    db, _, _ = db_with_fault(fault_type)
    assert faulted_kinds(db) == {fault_type.value}


def test_violations_are_hashable_values():
    assert {Violation("a", "1"), Violation("a", "1")} == {Violation("a", "1")}


def test_payment_tolerance_is_exactly_one_cent(db_with_fault):
    db, dataset, _ = db_with_fault(FaultType.UNBALANCED_JOURNAL)
    payment = dataset.tables["APPAYM"][0]
    update = 'UPDATE "APPAYM" SET "PAY_AMT" = "PAY_AMT" + %s WHERE "PAY_ID" = %s'
    assert PAYMENT_TOLERANCE == Decimal("0.01")
    with connect_admin(db) as conn, conn.transaction(force_rollback=True):
        before = global_violations(conn)
        conn.execute(update, (Decimal("0.01"), payment["PAY_ID"]))
        assert global_violations(conn) == before
        conn.execute(update, (Decimal("0.01"), payment["PAY_ID"]))
        new = global_violations(conn) - before
    assert new == {Violation("payment_exceeds_invoice", str(payment["INV_ID"]))}


def test_corrected_closed_posting_no_longer_counts(db_with_fault):
    db, _, fault = db_with_fault(FaultType.CLOSED_PERIOD_POSTING)
    assert run(db, fault, correct_fix(fault)).passed
    # verify rolled back, so the fault is still there.
    assert faulted_kinds(db) == {fault.type.value}


# Structure rules added with the resolution checks

@pytest.mark.parametrize("fault_type", [t for t in FaultType
                                        if t is not FaultType.DUPLICATE_INVOICE])
def test_status_change_is_refused_outside_a_duplicate_finding(fault_type, loaded):
    db, _, answers = loaded(spec({fault_type: 1, FaultType.DUPLICATE_INVOICE: 1}))
    fault = next(f for f in answers.faults if f.type is fault_type)
    duplicate = next(f for f in answers.faults if f.type is FaultType.DUPLICATE_INVOICE)
    fix = correction(None, status=duplicate.expected_status)
    assert_rejected(db, fault, fix, "structure", "status")


def test_void_cannot_hide_behind_an_existing_mismatch(loaded):
    """The void adds no new violation, so collateral alone would let it through."""
    for seed in range(1, 21):
        db, _, answers = loaded(spec({FaultType.UNBALANCED_JOURNAL: 1,
                                      FaultType.AP_GL_MISMATCH: 1}, seed=seed))
        fault, mismatch = answers.faults
        invoice = next(r.key for r in mismatch.rows if r.table == "APINVH")
        with connect_rw(db) as conn:
            stat_cd = conn.execute('SELECT "STAT_CD" FROM "APINVH" WHERE "INV_ID" = %s',
                                   (invoice,)).fetchone()[0]
        if stat_cd == "O":
            fix = correction(journal(fault), true_up(fault), [invoice])
            assert_rejected(db, fault, fix, "structure", "status")
            return
    pytest.fail("no seed planted a mismatch on an open invoice")


def test_manual_journal_cannot_post_to_the_ap_account(loaded):
    db, _, fault = find_unbalanced(loaded, on_invoice=False)
    fix = correction(journal(fault), [ap_plug(fault)])
    assert_rejected(db, fault, fix, "structure", "AP-sourced")


def test_invoice_journal_may_post_to_the_ap_account(db_with_fault):
    db, _, fault = db_with_fault(FaultType.AP_GL_MISMATCH)
    assert any(ln.acct_cd == code(AP_CONTROL) for ln in true_up(fault))
    assert run(db, fault, correct_fix(fault)).passed


def test_rejects_voiding_an_invoice_that_is_already_void(db_with_fault):
    db, _, fault = db_with_fault(FaultType.DUPLICATE_INVOICE)
    copy = next(iter(fault.expected_status))
    with connect_admin(db) as conn:
        conn.execute('UPDATE "APINVH" SET "STAT_CD" = %s WHERE "INV_ID" = %s', ("V", copy))
        try:
            assert_rejected(db, fault, correct_fix(fault), "structure", "already void")
        finally:
            conn.execute('UPDATE "APINVH" SET "STAT_CD" = %s WHERE "INV_ID" = %s', ("O", copy))


def bad_account_net(fault):
    bad = next(a for a, v in fault.expected_net.items() if v == 0)
    held = sum(k.dr_amt - k.cr_amt for k in fault.original_lines if k.acct_cd == bad)
    return bad, held


def test_rejects_bad_account_on_the_side_that_grows_its_net(db_with_fault):
    db, _, fault = db_with_fault(FaultType.INVALID_ACCOUNT)
    bad, held = bad_account_net(fault)
    assert held > 0
    fix = correction(journal(fault), [debit(bad, "1.00"), credit("5100", "1.00")])
    assert_rejected(db, fault, fix, "structure", bad.rstrip())


def test_rejects_taking_a_bad_account_past_zero(db_with_fault):
    db, _, fault = db_with_fault(FaultType.INVALID_ACCOUNT)
    bad, held = bad_account_net(fault)
    over = held + Decimal("0.01")
    fix = correction(journal(fault), [credit(bad, over), debit("5100", over)])
    assert_rejected(db, fault, fix, "structure", bad.rstrip())


def test_rejects_passing_a_bad_account_past_zero_across_several_lines(db_with_fault):
    db, _, fault = db_with_fault(FaultType.INVALID_ACCOUNT)
    bad, held = bad_account_net(fault)
    fix = correction(journal(fault), [credit(bad, held), credit(bad, "0.01"),
                                      debit("5100", held + Decimal("0.01"))])
    assert_rejected(db, fault, fix, "structure", bad.rstrip())


def test_rejects_a_bad_account_whose_group_already_nets_to_zero(db_with_fault):
    db, _, fault = db_with_fault(FaultType.INVALID_ACCOUNT)
    bad, held = bad_account_net(fault)
    with connect_admin(db) as conn:
        conn.execute('INSERT INTO "GLJRNH" ("JRNL_NO", "PRD_ID", "POST_DT", "POST_FLG", '
                     '"REF_JRNL_NO", "SRC_CD") VALUES (900001, %s, 20241101, %s, %s, %s)',
                     (OPEN_PRD, "P", journal(fault), "GL"))
        conn.execute('INSERT INTO "GLJRNL" VALUES (900001, 1, %s, 0, %s)', (bad, held))
    try:
        fix = correction(journal(fault), [credit(bad, "1.00"), debit("5100", "1.00")])
        assert_rejected(db, fault, fix, "structure", bad.rstrip())
    finally:
        with connect_admin(db) as conn:
            conn.execute('DELETE FROM "GLJRNL" WHERE "JRNL_NO" = 900001')
            conn.execute('DELETE FROM "GLJRNH" WHERE "JRNL_NO" = 900001')


@pytest.mark.parametrize("acct", [INACTIVE_ACCT, MISSING_ACCT])
def test_rejects_a_bad_account_absent_from_the_group(db_with_fault, acct):
    db, _, fault = db_with_fault(FaultType.AP_GL_MISMATCH)
    fix = correction(journal(fault), [debit(acct, "1.00"), credit("5100", "1.00")])
    assert_rejected(db, fault, fix, "structure", acct)


# Boundaries

def test_verifier_does_not_import_planting_code():
    package = Path(ledger_agent.verifier.__file__).parent
    for path in package.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            assert not any(n.startswith("ledger_agent.erp") or n.startswith("ledger_evals")
                           for n in names), path.name
    assert "PlantedFault" not in "".join(
        p.read_text(encoding="utf-8") for p in package.glob("*.py"))


def test_ap_account_matches_the_chart():
    assert code(AP_ACCOUNT) == code(AP_CONTROL)
