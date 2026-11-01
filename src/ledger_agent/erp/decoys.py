"""Decoys: rows that look wrong but are valid, used to measure false alarms."""

from ledger_agent.erp.generate import AP_CONTROL, N_PERIODS, YEAR, money, ymd
from ledger_agent.erp.plant import Planter, cents, inv_ref, jrnl_ref
from ledger_agent.models import DecoyType, PlantedDecoy, RowRef

RENT_ACCOUNT = "5300"
ROUNDING_CENTS = 1


def _reversing_entry(p: Planter) -> list[RowRef]:
    """An accrual reversed in full in the next period, referencing the original."""
    original = p.rng.choice([h for h in p.free_manual_journals()
                             if h["PRD_ID"] < N_PERIODS])
    jn = original["JRNL_NO"]
    prd = original["PRD_ID"] + 1
    legs = [(acct, cr, dr) for acct, dr, cr in p.legs(jn)]
    reversal = p.add_journal(prd, ymd(YEAR, prd, 1), "GL", original["SRC_REF"],
                             f"Reversal of accrual {jn}", legs, ref_jrnl=jn)
    return [jrnl_ref(reversal), jrnl_ref(jn)]


def _credit_memo(p: Planter) -> list[RowRef]:
    """A vendor credit for part of an invoice: a negative invoice whose journal
    debits AP and credits the original expense account."""
    original = p.rng.choice([i for i in p.single_expense_invoices()
                             if p.invoice_period(i) < N_PERIODS])
    expense = next(acct for acct, dr, _ in p.legs(original["JRNL_NO"]) if dr)
    credit = p.rng.randint(cents(original["INV_AMT"]) // 10, cents(original["INV_AMT"]) // 2)
    prd = p.invoice_period(original) + 1
    day = p.rng.randint(1, 20)
    inv_no = f"{original['INV_NO']}-CM"
    jn = p.add_journal(prd, ymd(YEAR, prd, day), "AP", inv_no, f"Credit memo {inv_no}",
                       [(AP_CONTROL, credit, 0), (expense, 0, credit)])
    inv_id = p.add_invoice(original["VEND_CD"], inv_no, ymd(YEAR, prd, day),
                           money(-credit), jn)
    return [inv_ref(inv_id), jrnl_ref(jn), inv_ref(original["INV_ID"]),
            jrnl_ref(original["JRNL_NO"])]


def _recurring_invoice(p: Planter) -> list[RowRef]:
    """Two identical monthly rent invoices, with distinct numbers, a month apart."""
    vend_cd = p.rng.choice(p.tables["APVEND"])["VEND_CD"]
    amount = p.rng.randint(100_000, 900_000)
    first = p.rng.randint(1, N_PERIODS - 1)
    rows = []
    for prd in (first, first + 1):
        inv_no = f"{vend_cd.strip()}-{p.rng.randint(10000, 99999)}-{p.next_inv}"
        dt = ymd(YEAR, prd, 1)
        jn = p.add_journal(prd, dt, "AP", inv_no, f"Monthly rent {inv_no}",
                           [(RENT_ACCOUNT, amount, 0), (AP_CONTROL, 0, amount)])
        rows += [inv_ref(p.add_invoice(vend_cd, inv_no, dt, money(amount), jn)), jrnl_ref(jn)]
    return rows


def _rounding_difference(p: Planter) -> list[RowRef]:
    """A payment one cent off its invoice, with its journal following the payment
    amount, so the ledger still balances and the cent stays open against AP."""
    invoices = {i["INV_ID"]: i for i in p.free_invoices()}
    payment = p.rng.choice([x for x in p.tables["APPAYM"] if x["INV_ID"] in invoices])
    delta = money(p.rng.choice((-ROUNDING_CENTS, ROUNDING_CENTS)))
    journal = p.payment_journal(payment)
    payment["PAY_AMT"] += delta
    for line in p.lines[journal["JRNL_NO"]]:
        line["DR_AMT" if line["DR_AMT"] else "CR_AMT"] += delta
    invoice = invoices[payment["INV_ID"]]
    return [inv_ref(invoice["INV_ID"]), jrnl_ref(journal["JRNL_NO"]),
            jrnl_ref(invoice["JRNL_NO"])]


_PLANTERS = {
    DecoyType.REVERSING_ENTRY: _reversing_entry,
    DecoyType.CREDIT_MEMO: _credit_memo,
    DecoyType.RECURRING_INVOICE: _recurring_invoice,
    DecoyType.ROUNDING_DIFFERENCE: _rounding_difference,
}


def plant_decoy(p: Planter, decoy_id: str, decoy_type: DecoyType) -> PlantedDecoy:
    return PlantedDecoy(decoy_id=decoy_id, type=decoy_type,
                        rows=p.claim(_PLANTERS[decoy_type](p)))
