"""Shared data models for planted scenarios and their answer key."""

from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


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

    expected_status maps INV_ID to the STAT_CD a correction must set.
    """

    fault_id: str
    type: FaultType
    rows: list[RowRef]
    original_lines: list[KeyLine]
    expected_net: dict[str, Decimal]
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
