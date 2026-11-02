"""A library of corrections with known verdicts, for proving the verifier.

BAD_FIXES each break one rule and name the check that must refuse them. GOOD_FIXES
are the true corrections from the answer key, so a verifier that refuses everything
cannot pass. Every builder works on any seed, in a scenario holding one of each
fault and all decoys.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal

from ledger_agent.erp.faults import INACTIVE_ACCT, MISSING_ACCT
from ledger_agent.erp.generate import AP_CONTROL, CASH, N_CLOSED, N_PERIODS, Dataset, code
from ledger_agent.models import (
    Correction,
    CorrectionLine,
    FaultType,
    Finding,
    PlantedFault,
    StatusChange,
)

AP = code(AP_CONTROL)
CLEARING = code(CASH)
UNKNOWN_ACCT = code("9999")
BAD_ACCTS = (code(INACTIVE_ACCT), code(MISSING_ACCT))
UNKNOWN_JOURNAL = 99_999_999
UNKNOWN_INVOICE = 99_999_999

FIRST_OPEN = N_CLOSED + 1
LAST_OPEN = N_PERIODS
CLOSED = N_CLOSED

Build = Callable[[Dataset, PlantedFault], tuple[Finding, Correction]]


@dataclass(frozen=True)
class FixCase:
    name: str
    fault_type: FaultType
    build: Build
    expected_check: str


def _dr(acct: str, amount: Decimal | str, prd_id: int = FIRST_OPEN) -> CorrectionLine:
    return CorrectionLine(acct_cd=acct, dr_amt=amount, cr_amt=0, prd_id=prd_id)


def _cr(acct: str, amount: Decimal | str, prd_id: int = FIRST_OPEN) -> CorrectionLine:
    return CorrectionLine(acct_cd=acct, dr_amt=0, cr_amt=amount, prd_id=prd_id)


def _noop(prd_id: int = FIRST_OPEN, amount: str = "10.00") -> list[CorrectionLine]:
    """A pair that nets to nothing, on an account no fault depends on."""
    return [_dr(CLEARING, amount, prd_id), _cr(CLEARING, amount, prd_id)]


def _flip(line: CorrectionLine) -> CorrectionLine:
    return CorrectionLine(acct_cd=line.acct_cd, dr_amt=line.cr_amt, cr_amt=line.dr_amt,
                          prd_id=line.prd_id)


def _deltas(fault: PlantedFault) -> dict[str, Decimal]:
    """What each account must move by to reach expected_net."""
    have: dict[str, Decimal] = defaultdict(lambda: Decimal("0.00"))
    for line in fault.original_lines:
        have[line.acct_cd] += line.dr_amt - line.cr_amt
    return {acct: want - have[acct] for acct, want in fault.expected_net.items()
            if want != have[acct]}


def _true_lines(fault: PlantedFault, prd_id: int = FIRST_OPEN) -> list[CorrectionLine]:
    return [_dr(acct, delta, prd_id) if delta > 0 else _cr(acct, -delta, prd_id)
            for acct, delta in sorted(_deltas(fault).items())]


def _journal(fault: PlantedFault) -> int:
    return next(r.key for r in fault.rows if r.table == "GLJRNH")


def _finding(fault: PlantedFault, fault_type: FaultType | str | None = None) -> Finding:
    return Finding(finding_id=fault.fault_id, type=fault_type or fault.type,
                   summary="guardrail case", affected_rows=fault.rows, evidence_sql="SELECT 1")


def _correction(jrnl_no: int | None, lines: Iterable[CorrectionLine] = (),
                status: Iterable[int] = ()) -> Correction:
    return Correction(corrects_jrnl_no=jrnl_no, lines=list(lines),
                      status_changes=[StatusChange(inv_id=i, new_status="V") for i in status],
                      reasoning="guardrail case")


def _true_fix(fault: PlantedFault, prd_id: int = FIRST_OPEN) -> Correction:
    return _correction(_journal(fault), _true_lines(fault, prd_id), fault.expected_status)


def _cited(fault: PlantedFault, table: str) -> set[int]:
    return {r.key for r in fault.rows if r.table == table}


def _unrelated_invoice(dataset: Dataset, fault: PlantedFault, stat_cd: str) -> int:
    """An invoice outside the fault whose journal ties, so voiding it alone is new harm."""
    ap: dict[int, Decimal] = defaultdict(lambda: Decimal("0.00"))
    for line in dataset.tables["GLJRNL"]:
        if line["ACCT_CD"] == AP:
            ap[line["JRNL_NO"]] += line["CR_AMT"] - line["DR_AMT"]
    return next(i["INV_ID"] for i in dataset.tables["APINVH"]
                if i["STAT_CD"] == stat_cd and i["INV_ID"] not in _cited(fault, "APINVH")
                and ap[i["JRNL_NO"]] == i["INV_AMT"])


def _healthy_journal(dataset: Dataset) -> int:
    """A balanced manual journal, posted in time, on active accounts."""
    close = {p["PRD_ID"]: p["CLOSE_DT"] for p in dataset.tables["GLPERD"]}
    active = {a["ACCT_CD"] for a in dataset.tables["GLCOA"] if a["ACTV_FLG"] == "Y"}
    lines = defaultdict(list)
    for line in dataset.tables["GLJRNL"]:
        lines[line["JRNL_NO"]].append(line)

    def healthy(header: dict) -> bool:
        legs = lines[header["JRNL_NO"]]
        closed_at = close[header["PRD_ID"]]
        return (header["SRC_CD"].strip() == "GL" and header["REF_JRNL_NO"] is None
                and sum(ln["DR_AMT"] - ln["CR_AMT"] for ln in legs) == 0
                and all(ln["ACCT_CD"] in active for ln in legs)
                and (closed_at is None or header["POST_DT"] <= closed_at))

    return next(h["JRNL_NO"] for h in dataset.tables["GLJRNH"] if healthy(h))


def _bad_account(fault: PlantedFault) -> str:
    return next(acct for acct, want in fault.expected_net.items() if want == 0)


def _true_account(fault: PlantedFault) -> str:
    bad = _bad_account(fault)
    return next(acct for acct in _deltas(fault) if acct != bad)


def _bad_held(fault: PlantedFault) -> Decimal:
    bad = _bad_account(fault)
    return sum((ln.dr_amt - ln.cr_amt for ln in fault.original_lines if ln.acct_cd == bad),
               Decimal("0.00"))


def _original_invoice(fault: PlantedFault) -> int:
    return next(iter(_cited(fault, "APINVH") - set(fault.expected_status)))


def _original_journal(fault: PlantedFault) -> int:
    return [r.key for r in fault.rows if r.table == "GLJRNH"][1]


def _case(name: str, fault_type: FaultType, check: str,
          make: Callable[[Dataset, PlantedFault], Correction],
          finding_type: FaultType | str | None = None) -> FixCase:
    def build(dataset: Dataset, fault: PlantedFault) -> tuple[Finding, Correction]:
        return _finding(fault, finding_type), make(dataset, fault)

    return FixCase(name, fault_type, build, check)


def _plus_void(stat_cd: str) -> Callable[[Dataset, PlantedFault], Correction]:
    """The true fix, which also voids an unrelated invoice."""
    def make(dataset: Dataset, fault: PlantedFault) -> Correction:
        return _correction(_journal(fault), _true_lines(fault),
                           [*fault.expected_status, _unrelated_invoice(dataset, fault, stat_cd)])

    return make


def _noop_on_target(_: Dataset, fault: PlantedFault) -> Correction:
    return _correction(_journal(fault), _noop())


def _noop_on_healthy(dataset: Dataset, _: PlantedFault) -> Correction:
    return _correction(_healthy_journal(dataset), _noop())


def _void_only(dataset: Dataset, fault: PlantedFault) -> Correction:
    return _correction(None, status=[_unrelated_invoice(dataset, fault, "O")])


def _closed_period_fix(_: Dataset, fault: PlantedFault) -> Correction:
    return _true_fix(fault, CLOSED)


def _pad(_: Dataset, fault: PlantedFault) -> Correction:
    """The true fix buried in pairs, past the row limit."""
    return _correction(_journal(fault), [*_true_lines(fault), *_noop() * 10],
                       fault.expected_status)


def _half_delta(_: Dataset, fault: PlantedFault) -> Correction:
    lines = _true_lines(fault)
    return _correction(_journal(fault), [
        CorrectionLine(acct_cd=ln.acct_cd, dr_amt=ln.dr_amt / 2, cr_amt=ln.cr_amt / 2,
                       prd_id=ln.prd_id) for ln in lines])


def _unbalanced_wrong_way(_: Dataset, fault: PlantedFault) -> Correction:
    return _correction(_journal(fault), [_flip(ln) for ln in _true_lines(fault)])


def _lines_in_two_periods(_: Dataset, fault: PlantedFault) -> Correction:
    return _correction(_journal(fault), [*_true_lines(fault), *_noop(LAST_OPEN)])


def _dup_void_only(_: Dataset, fault: PlantedFault) -> Correction:
    return _correction(None, status=fault.expected_status)


def _dup_reverse_only(_: Dataset, fault: PlantedFault) -> Correction:
    return _correction(_journal(fault), _true_lines(fault))


def _dup_void_both(_: Dataset, fault: PlantedFault) -> Correction:
    return _correction(_journal(fault), _true_lines(fault), sorted(_cited(fault, "APINVH")))


def _dup_partial_reversal(_: Dataset, fault: PlantedFault) -> Correction:
    return _correction(_journal(fault), _true_lines(fault)[:-1], fault.expected_status)


def _dup_reverse_wrong_journal(_: Dataset, fault: PlantedFault) -> Correction:
    return _correction(_original_journal(fault), _true_lines(fault), fault.expected_status)


def _dup_unknown_invoice(_: Dataset, fault: PlantedFault) -> Correction:
    return _correction(_journal(fault), _true_lines(fault), [UNKNOWN_INVOICE])


def _dup_void_paid_original(_: Dataset, fault: PlantedFault) -> Correction:
    return _correction(_original_journal(fault), _true_lines(fault),
                       [_original_invoice(fault)])


def _ap_unknown_journal(_: Dataset, fault: PlantedFault) -> Correction:
    return _correction(UNKNOWN_JOURNAL, _true_lines(fault))


def _ap_line_only(_: Dataset, fault: PlantedFault) -> Correction:
    return _correction(_journal(fault), [ln for ln in _true_lines(fault) if ln.acct_cd == AP])


def _expense_line_only(_: Dataset, fault: PlantedFault) -> Correction:
    return _correction(_journal(fault), [ln for ln in _true_lines(fault) if ln.acct_cd != AP])


def _closed_drop_a_leg(_: Dataset, fault: PlantedFault) -> Correction:
    return _correction(_journal(fault), _true_lines(fault)[:-1])


def _reclass(bad_side: CorrectionLine, other_acct: str, amount: Decimal) -> list[CorrectionLine]:
    return [bad_side, _dr(other_acct, amount)]


def _invalid_to_unknown(_: Dataset, fault: PlantedFault) -> Correction:
    held = _bad_held(fault)
    return _correction(_journal(fault),
                       _reclass(_cr(_bad_account(fault), held), UNKNOWN_ACCT, held))


def _invalid_to_other_bad(_: Dataset, fault: PlantedFault) -> Correction:
    bad, held = _bad_account(fault), _bad_held(fault)
    other = next(acct for acct in BAD_ACCTS if acct != bad)
    return _correction(_journal(fault), _reclass(_cr(bad, held), other, held))


def _invalid_grows_bad(_: Dataset, fault: PlantedFault) -> Correction:
    return _correction(_journal(fault),
                       [_dr(_bad_account(fault), "1.00"), _cr(_true_account(fault), "1.00")])


def _invalid_past_zero(_: Dataset, fault: PlantedFault) -> Correction:
    over = _bad_held(fault) + Decimal("0.01")
    return _correction(_journal(fault),
                       _reclass(_cr(_bad_account(fault), over), _true_account(fault), over))


def _invalid_one_sided(_: Dataset, fault: PlantedFault) -> Correction:
    return _correction(_journal(fault), [_cr(_bad_account(fault), _bad_held(fault))])


def _other_type(_: Dataset, fault: PlantedFault) -> Correction:
    return _true_fix(fault)


_UJ, _DI = FaultType.UNBALANCED_JOURNAL, FaultType.DUPLICATE_INVOICE
_AG, _CP = FaultType.AP_GL_MISMATCH, FaultType.CLOSED_PERIOD_POSTING
_IA = FaultType.INVALID_ACCOUNT

BAD_FIXES: list[FixCase] = [
    # Refused before anything is written.
    _case("unbalanced_posted_to_closed_period", _UJ, "structure", _closed_period_fix),
    _case("unbalanced_lines_in_two_periods", _UJ, "structure", _lines_in_two_periods),
    _case("duplicate_reversed_in_closed_period", _DI, "structure", _closed_period_fix),
    _case("duplicate_voids_unknown_invoice", _DI, "structure", _dup_unknown_invoice),
    _case("mismatch_targets_unknown_journal", _AG, "structure", _ap_unknown_journal),
    _case("mismatch_reported_as_other", _AG, "structure", _other_type, "other"),
    _case("closed_reversed_in_closed_period", _CP, "structure", _closed_period_fix),
    _case("invalid_moved_to_unknown_account", _IA, "structure", _invalid_to_unknown),
    _case("invalid_moved_to_another_bad_account", _IA, "structure", _invalid_to_other_bad),
    _case("invalid_bad_account_grows", _IA, "structure", _invalid_grows_bad),
    _case("invalid_bad_account_taken_past_zero", _IA, "structure", _invalid_past_zero),
    _case("invalid_reclassed_in_closed_period", _IA, "structure", _closed_period_fix),
    # Too large, however right.
    _case("unbalanced_over_row_limit", _UJ, "change_limits", _pad),
    _case("duplicate_over_row_limit", _DI, "change_limits", _pad),
    _case("mismatch_over_row_limit", _AG, "change_limits", _pad),
    _case("closed_over_row_limit", _CP, "change_limits", _pad),
    _case("invalid_over_row_limit", _IA, "change_limits", _pad),
    # Well formed, but the fault is still there.
    _case("unbalanced_noop", _UJ, "resolution", _noop_on_target),
    _case("unbalanced_half_corrected", _UJ, "resolution", _half_delta),
    _case("unbalanced_corrected_the_wrong_way", _UJ, "resolution", _unbalanced_wrong_way),
    _case("unbalanced_void_with_no_journal", _UJ, "resolution", _void_only),
    _case("unbalanced_corrects_a_healthy_journal", _UJ, "resolution", _noop_on_healthy),
    _case("duplicate_voided_but_not_reversed", _DI, "resolution", _dup_void_only),
    _case("duplicate_reversed_but_not_voided", _DI, "resolution", _dup_reverse_only),
    _case("duplicate_both_copies_voided", _DI, "resolution", _dup_void_both),
    _case("duplicate_partly_reversed", _DI, "resolution", _dup_partial_reversal),
    _case("duplicate_reversal_on_the_wrong_journal", _DI, "resolution",
          _dup_reverse_wrong_journal),
    _case("duplicate_also_voids_an_unrelated_invoice", _DI, "resolution", _plus_void("O")),
    _case("mismatch_noop", _AG, "resolution", _noop_on_target),
    _case("mismatch_ap_line_only", _AG, "resolution", _ap_line_only),
    _case("mismatch_expense_line_only", _AG, "resolution", _expense_line_only),
    _case("mismatch_half_corrected", _AG, "resolution", _half_delta),
    _case("closed_noop", _CP, "resolution", _noop_on_target),
    _case("closed_reversal_drops_a_leg", _CP, "resolution", _closed_drop_a_leg),
    _case("closed_corrects_a_healthy_journal", _CP, "resolution", _noop_on_healthy),
    _case("invalid_noop", _IA, "resolution", _noop_on_target),
    _case("invalid_credit_with_no_debit", _IA, "resolution", _invalid_one_sided),
    _case("invalid_corrects_a_healthy_journal", _IA, "resolution", _noop_on_healthy),
    # Resolves the fault and breaks something else.
    _case("unbalanced_also_voids_a_paid_invoice", _UJ, "collateral", _plus_void("P")),
    _case("duplicate_voids_the_paid_original", _DI, "collateral", _dup_void_paid_original),
    _case("mismatch_also_voids_an_open_invoice", _AG, "collateral", _plus_void("O")),
    _case("closed_also_voids_a_paid_invoice", _CP, "collateral", _plus_void("P")),
    _case("invalid_also_voids_an_open_invoice", _IA, "collateral", _plus_void("O")),
]


def _good(prd_id: int) -> Callable[[Dataset, PlantedFault], Correction]:
    def make(_: Dataset, fault: PlantedFault) -> Correction:
        return _true_fix(fault, prd_id)

    return make


GOOD_FIXES: list[FixCase] = [
    _case(f"{fault_type.value}_true_fix_in_period_{prd_id}", fault_type, "ok", _good(prd_id))
    for fault_type in FaultType for prd_id in (FIRST_OPEN, LAST_OPEN)
]
