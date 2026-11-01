"""Shared mutable view of a Dataset for planting faults and decoys.

Every planted item claims its rows, and candidates exclude claimed rows, so no
two items in a scenario share a row. Amounts are integer cents here, as in
generate.py.
"""

import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from decimal import Decimal

from ledger_agent.erp.generate import Dataset, code, money
from ledger_agent.models import RowRef


def cents(amount: Decimal) -> int:
    return int(amount.scaleb(2))


def jrnl_ref(jrnl_no: int) -> RowRef:
    return RowRef(table="GLJRNH", key=jrnl_no)


def inv_ref(inv_id: int) -> RowRef:
    return RowRef(table="APINVH", key=inv_id)


class Planter:
    def __init__(self, dataset: Dataset, rng: random.Random):
        self.tables = dataset.tables
        self.rng = rng
        self.claimed: set[RowRef] = set()
        self.periods: set[int] = set()
        self.headers = {h["JRNL_NO"]: h for h in self.tables["GLJRNH"]}
        self.lines: dict[int, list[dict]] = defaultdict(list)
        for line in self.tables["GLJRNL"]:
            self.lines[line["JRNL_NO"]].append(line)
        self.next_jrnl = max(self.headers) + 1
        self.next_inv = max(i["INV_ID"] for i in self.tables["APINVH"]) + 1

    def claim(self, rows: Iterable[RowRef]) -> list[RowRef]:
        rows = list(rows)
        taken = self.claimed.intersection(rows)
        if taken or len(set(rows)) != len(rows):
            raise ValueError(f"rows already planted: {sorted(taken, key=str)}")
        self.claimed.update(rows)
        return rows

    def is_free(self, *rows: RowRef) -> bool:
        return self.claimed.isdisjoint(rows)

    def pick_period_varied[T](self, candidates: Sequence[T], period: Callable[[T], int]) -> T:
        """Choose a candidate, preferring a period no earlier fault has used."""
        fresh = [c for c in candidates if period(c) not in self.periods]
        chosen = self.rng.choice(fresh or candidates)
        self.periods.add(period(chosen))
        return chosen

    def invoice_period(self, invoice: dict) -> int:
        return self.headers[invoice["JRNL_NO"]]["PRD_ID"]

    def free_invoices(self) -> list[dict]:
        return [i for i in self.tables["APINVH"]
                if self.is_free(inv_ref(i["INV_ID"]), jrnl_ref(i["JRNL_NO"]))]

    def single_expense_invoices(self) -> list[dict]:
        """Free invoices whose journal is one expense line and one AP line, so the
        invoice total alone fixes what each line should have been."""
        return [i for i in self.free_invoices() if len(self.lines[i["JRNL_NO"]]) == 2]

    def free_manual_journals(self) -> list[dict]:
        return [h for h in self.tables["GLJRNH"]
                if h["SRC_CD"] == "GL" and h["REF_JRNL_NO"] is None
                and self.is_free(jrnl_ref(h["JRNL_NO"]))]

    def payment_journal(self, payment: dict) -> dict:
        ref = f"PAY-{payment['PAY_ID']}"
        return next(h for h in self.tables["GLJRNH"] if h["SRC_REF"] == ref)

    def add_journal(self, prd: int, post_dt: int, src: str, ref: str, memo: str,
                    legs: Sequence[tuple[str, int, int]],
                    ref_jrnl: int | None = None) -> int:
        """Append a posted journal. legs are (padded account, debit, credit)."""
        jn = self.next_jrnl
        self.next_jrnl += 1
        header = {"JRNL_NO": jn, "PRD_ID": prd, "POST_DT": post_dt, "POST_FLG": "P",
                  "REF_JRNL_NO": ref_jrnl, "SRC_CD": src, "SRC_REF": ref, "MEMO": memo}
        self.tables["GLJRNH"].append(header)
        self.headers[jn] = header
        for seq, (acct, dr, cr) in enumerate(legs, start=1):
            line = {"JRNL_NO": jn, "LN_SEQ": seq, "ACCT_CD": code(acct.rstrip()),
                    "DR_AMT": money(dr), "CR_AMT": money(cr)}
            self.tables["GLJRNL"].append(line)
            self.lines[jn].append(line)
        return jn

    def add_invoice(self, vend_cd: str, inv_no: str, inv_dt: int, amount: Decimal,
                    jrnl_no: int) -> int:
        """Append an unpaid invoice, returning its INV_ID."""
        inv_id = self.next_inv
        self.next_inv += 1
        self.tables["APINVH"].append({
            "INV_ID": inv_id, "VEND_CD": vend_cd, "INV_NO": inv_no, "INV_DT": inv_dt,
            "INV_AMT": amount, "STAT_CD": "O", "JRNL_NO": jrnl_no})
        return inv_id

    def legs(self, jrnl_no: int) -> list[tuple[str, int, int]]:
        return [(ln["ACCT_CD"], cents(ln["DR_AMT"]), cents(ln["CR_AMT"]))
                for ln in self.lines[jrnl_no]]
