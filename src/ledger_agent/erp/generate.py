"""Seeded generator for the clean legacy ERP dataset.

Pure Python and deterministic: one random.Random(seed), fixed base dates, no
clock. Rows are plain dicts keyed by column name so later steps can mutate them.
"""

import calendar
import random
from dataclasses import dataclass, field
from decimal import Decimal

YEAR = 2024
N_PERIODS = 12
N_CLOSED = 10
N_VENDORS = 24
N_INVOICES = 800
N_MANUAL_JOURNALS = 150
CLOSE_LAG_DAYS = 5

AP_CONTROL = "2000"
CASH = "1000"
EXPENSE_ACCOUNTS = ["5100", "5200", "5300", "5400", "5500"]

COA = [
    ("1000", "Operating Cash", "ASSET", "Y"),
    ("1200", "Prepaid Expenses", "ASSET", "Y"),
    ("2000", "Accounts Payable Control", "LIABILITY", "Y"),
    ("2100", "Accrued Liabilities", "LIABILITY", "Y"),
    ("5100", "Office Supplies", "EXPENSE", "Y"),
    ("5200", "Professional Fees", "EXPENSE", "Y"),
    ("5300", "Facilities", "EXPENSE", "Y"),
    ("5400", "Software and Subscriptions", "EXPENSE", "Y"),
    ("5500", "Travel", "EXPENSE", "Y"),
    ("5900", "Legacy Miscellaneous", "EXPENSE", "N"),
]

VENDOR_WORDS = ["Apex", "Birch", "Cobalt", "Delta", "Ember", "Falcon", "Granite",
                "Harbor", "Iron", "Juniper", "Keystone", "Lumen", "Meridian",
                "Northwind", "Orchard", "Pioneer", "Quartz", "Ridge", "Summit",
                "Tidewater", "Union", "Vertex", "Willow", "Yarrow"]
VENDOR_KINDS = ["Supply", "Partners", "Services", "Systems", "Group"]
USERS = ["JSMITH", "MPATEL", "RLOPEZ", "KWONG"]


@dataclass
class Dataset:
    tables: dict[str, list[dict]] = field(default_factory=dict)


def code(value: str) -> str:
    return value.ljust(8)


def money(cents: int) -> Decimal:
    return Decimal(cents).scaleb(-2)


def _ymd(year: int, month: int, day: int) -> int:
    return year * 10000 + month * 100 + day


def _periods() -> list[dict]:
    rows = []
    for m in range(1, N_PERIODS + 1):
        last = calendar.monthrange(YEAR, m)[1]
        closed = m <= N_CLOSED
        close = None
        if closed:
            # Close happens CLOSE_LAG_DAYS after month end, always in the next month.
            close = _ymd(YEAR if m < 12 else YEAR + 1, m % 12 + 1, CLOSE_LAG_DAYS)
        rows.append({"PRD_ID": m, "STRT_DT": _ymd(YEAR, m, 1),
                     "END_DT": _ymd(YEAR, m, last), "CLOSE_DT": close,
                     "STAT_CD": "C" if closed else "O"})
    return rows


def generate_clean(seed: int) -> Dataset:
    rng = random.Random(seed)

    coa = [{"ACCT_CD": code(a), "ACCT_NM": n, "ACCT_TYP": t, "ACTV_FLG": f}
           for a, n, t, f in COA]

    names = [f"{w} {rng.choice(VENDOR_KINDS)}" for w in VENDOR_WORDS[:N_VENDORS]]
    vendors = [{"VEND_CD": code(f"V{i:04d}"), "VEND_NM": n, "ACTV_FLG": "Y"}
               for i, n in enumerate(names, start=1)]

    headers, lines, invoices, payments = [], [], [], []
    next_jrnl = 1000

    def add_journal(prd, day, src, ref, memo, legs):
        nonlocal next_jrnl
        jn = next_jrnl
        next_jrnl += 1
        headers.append({"JRNL_NO": jn, "PRD_ID": prd, "POST_DT": _ymd(YEAR, prd, day),
                        "POST_FLG": "P", "REF_JRNL_NO": None, "SRC_CD": src,
                        "SRC_REF": ref, "MEMO": memo})
        for seq, (acct, dr, cr) in enumerate(legs, start=1):
            lines.append({"JRNL_NO": jn, "LN_SEQ": seq, "ACCT_CD": code(acct),
                          "DR_AMT": money(dr), "CR_AMT": money(cr)})
        return jn

    def split(total_cents):
        """Split an amount across 1-2 expense accounts."""
        if rng.random() < 0.5:
            return [(rng.choice(EXPENSE_ACCOUNTS), total_cents)]
        a, b = rng.sample(EXPENSE_ACCOUNTS, 2)
        first = rng.randint(1, total_cents - 1)
        return [(a, first), (b, total_cents - first)]

    pay_id = 1
    for inv_id in range(1, N_INVOICES + 1):
        vend = rng.choice(vendors)
        prd = rng.randint(1, N_PERIODS)
        day = rng.randint(1, 20)
        total = rng.randint(5_000, 2_500_000)
        inv_no = f"{vend['VEND_CD'].strip()}-{rng.randint(10000, 99999)}-{inv_id}"
        legs = [(a, c, 0) for a, c in split(total)] + [(AP_CONTROL, 0, total)]
        jn = add_journal(prd, day, "AP", inv_no, f"Invoice {inv_no}", legs)
        inv_dt = _ymd(YEAR, prd, day)

        pay_dt = None
        if rng.random() < 0.7:
            pay_prd = prd + rng.randint(0, 2)
            if pay_prd <= N_PERIODS:
                pay_day = rng.randint(day + 1, 28)
                pay_dt = _ymd(YEAR, pay_prd, pay_day)
                pjn = add_journal(pay_prd, pay_day, "AP", f"PAY-{pay_id}",
                                  f"Payment {inv_no}",
                                  [(AP_CONTROL, total, 0), (CASH, 0, total)])
                payments.append({"PAY_ID": pay_id, "INV_ID": inv_id,
                                 "PAY_DT": pay_dt, "PAY_AMT": money(total)})
                pay_id += 1
        invoices.append({"INV_ID": inv_id, "VEND_CD": vend["VEND_CD"], "INV_NO": inv_no,
                         "INV_DT": inv_dt, "INV_AMT": money(total),
                         "STAT_CD": "P" if pay_dt else "O", "JRNL_NO": jn})

    for _ in range(N_MANUAL_JOURNALS):
        prd = rng.randint(1, N_PERIODS)
        total = rng.randint(10_000, 800_000)
        exp = rng.choice(EXPENSE_ACCOUNTS)
        legs = [(exp, total, 0), ("2100", 0, total // 2), ("1200", 0, total - total // 2)]
        add_journal(prd, rng.randint(1, 25), "GL", rng.choice(USERS), "Manual accrual", legs)

    headers.sort(key=lambda h: h["JRNL_NO"])
    return Dataset(tables={
        "GLPERD": _periods(), "GLCOA": coa, "GLJRNH": headers, "GLJRNL": lines,
        "APVEND": vendors, "APINVH": invoices, "APPAYM": payments,
    })
