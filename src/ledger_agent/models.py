"""Shared data models for planted scenarios and their answer key."""

from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# GLJRNL amounts are numeric(14,2).
MAX_AMOUNT = Decimal("999999999999.99")


class FaultType(StrEnum):
    UNBALANCED_JOURNAL = "unbalanced_journal"
    DUPLICATE_INVOICE = "duplicate_invoice"
    AP_GL_MISMATCH = "ap_gl_mismatch"
    CLOSED_PERIOD_POSTING = "closed_period_posting"
    INVALID_ACCOUNT = "invalid_account"


class DecoyType(StrEnum):
    REVERSING_ENTRY = "reversing_entry"
    CREDIT_MEMO = "credit_memo"
    RECURRING_INVOICE = "recurring_invoice"
    ROUNDING_DIFFERENCE = "rounding_difference"


class RowRef(BaseModel):
    """A row by table and integer key: JRNL_NO for GLJRNH, INV_ID for APINVH."""

    model_config = ConfigDict(frozen=True)

    table: str
    key: int


class KeyLine(BaseModel):
    """A journal line as stored: acct_cd keeps its space-padded 8-character form."""

    acct_cd: str
    dr_amt: Decimal
    cr_amt: Decimal


class PlantedFault(BaseModel):
    """One planted fault and what a valid correction must achieve.

    rows lists every row a reasonable finding might cite. Its first GLJRNH entry
    is the journal a correction targets.

    original_lines are that journal's lines as they stand in the faulted data.
    Those lines plus a correction's lines must net to expected_net by account.

    expected_net is debit minus credit per account. Keys are account codes in the
    stored space-padded 8-character form. It names every account in original_lines,
    with an explicit zero where the position must clear, and any account it omits
    must net to zero.

    expected_status maps INV_ID to the STAT_CD a correction must set, and is every
    status change a correction may make.

    open_periods lists the PRD_IDs a correction may post to.
    """

    fault_id: str
    type: FaultType
    rows: list[RowRef]
    original_lines: list[KeyLine]
    expected_net: dict[str, Decimal]
    open_periods: list[int]
    expected_status: dict[int, str] = {}


class PlantedDecoy(BaseModel):
    """Rows that look wrong but are valid. Citing one is a false alarm."""

    decoy_id: str
    type: DecoyType
    rows: list[RowRef]


class AnswerKey(BaseModel):
    seed: int
    faults: list[PlantedFault]
    decoys: list[PlantedDecoy]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)


class ScenarioSpec(BaseModel):
    """What to plant. faults maps each fault type to how many to plant."""

    id: str
    split: Literal["dev", "holdout", "injection"]
    seed: int
    faults: dict[FaultType, int]
    decoys: list[DecoyType]
    hostile_memo: bool = False


class Finding(BaseModel):
    """A problem the agent reports. Type other has no correction."""

    finding_id: str
    type: FaultType | Literal["other"]
    summary: str
    affected_rows: list[RowRef]
    evidence_sql: str


def _parse_amount(value: object) -> Decimal:
    """An exact non-negative amount at 2 decimal places. Floats go through str, so
    0.1 means 0.10. An amount that needs rounding is refused, never rounded."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError("amount must be a number")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as e:
        raise ValueError("amount must be a number") from e
    if not amount.is_finite():
        raise ValueError("amount must be finite")
    if amount < 0:
        raise ValueError("amount must not be negative")
    if amount > MAX_AMOUNT:
        raise ValueError(f"amount must not exceed {MAX_AMOUNT}")
    exact = amount.quantize(Decimal("0.01"))
    if exact != amount:
        raise ValueError("amount must have at most 2 decimal places")
    return abs(exact)


class CorrectionLine(BaseModel):
    """One journal line. acct_cd is normalised to the stored 8-character form."""

    acct_cd: str
    dr_amt: Decimal
    cr_amt: Decimal
    prd_id: int

    @field_validator("acct_cd")
    @classmethod
    def _pad_account(cls, value: str) -> str:
        return value.rstrip().ljust(8)

    @field_validator("dr_amt", "cr_amt", mode="before")
    @classmethod
    def _exact_amount(cls, value: object) -> Decimal:
        return _parse_amount(value)

    @model_validator(mode="after")
    def _one_side(self) -> "CorrectionLine":
        if self.dr_amt and self.cr_amt:
            raise ValueError("a line cannot have both a debit and a credit")
        if not self.dr_amt and not self.cr_amt:
            raise ValueError("a line needs a debit or a credit")
        return self


class StatusChange(BaseModel):
    inv_id: int
    new_status: Literal["V"]


class Correction(BaseModel):
    """A proposed fix: journal lines referencing corrects_jrnl_no, status changes,
    or both."""

    corrects_jrnl_no: int | None
    lines: list[CorrectionLine]
    status_changes: list[StatusChange]
    reasoning: str
