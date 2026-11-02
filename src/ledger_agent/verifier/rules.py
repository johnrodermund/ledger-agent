"""Whether a correction resolves its finding, judged after the trial run.

Rules read stored rows inside the trial transaction, so they judge what was
actually written and not what the correction says. Each returns a failure reason
or None.
"""

from collections.abc import Callable
from decimal import Decimal

import psycopg

from ledger_agent.models import Correction, FaultType, Finding
from ledger_agent.verifier.invariants import AP_ACCOUNT, group_net


def _imbalance(jrnl_no: int, net: dict[str, Decimal]) -> str | None:
    total = sum(net.values())
    return f"journal group {jrnl_no} does not balance: off by {total}" if total else None


def _left_open(jrnl_no: int, net: dict[str, Decimal]) -> str | None:
    """The first account the group has not netted to zero."""
    for acct, amount in sorted(net.items()):
        if amount:
            return f"journal group {jrnl_no} does not net to zero: " \
                   f"account {acct.rstrip()} carries {amount}"
    return None


def _unbalanced_journal(conn: psycopg.Connection, correction: Correction) -> str | None:
    jrnl_no = correction.corrects_jrnl_no
    return _imbalance(jrnl_no, group_net(conn, jrnl_no))


def _duplicate_invoice(conn: psycopg.Connection, correction: Correction) -> str | None:
    if not correction.status_changes:
        return "no invoice was voided"
    for change in correction.status_changes:
        vend_cd, inv_no, stat_cd, jrnl_no = conn.execute(
            'SELECT "VEND_CD", "INV_NO", "STAT_CD", "JRNL_NO" FROM "APINVH" '
            'WHERE "INV_ID" = %s', (change.inv_id,)).fetchone()
        if stat_cd != "V":
            return f"invoice {change.inv_id} was not voided"
        remaining = conn.execute(
            'SELECT count(*) FROM "APINVH" WHERE "VEND_CD" = %s '
            'AND trim("INV_NO") = trim(%s) AND "STAT_CD" <> %s',
            (vend_cd, inv_no, "V")).fetchone()[0]
        if remaining != 1:
            return f"{remaining} non-void invoices remain for vendor {vend_cd.rstrip()} " \
                   f"number {inv_no.strip()}, expected 1"
        problem = _left_open(jrnl_no, group_net(conn, jrnl_no))
        if problem:
            return problem
    return None


def _ap_gl_mismatch(conn: psycopg.Connection, correction: Correction) -> str | None:
    jrnl_no = correction.corrects_jrnl_no
    invoice = conn.execute('SELECT "INV_AMT" FROM "APINVH" WHERE "JRNL_NO" = %s',
                           (jrnl_no,)).fetchone()
    if invoice is None:
        return f"journal {jrnl_no} has no invoice"
    net = group_net(conn, jrnl_no)
    ap_total = -sum((amount for acct, amount in net.items() if acct.rstrip() == AP_ACCOUNT),
                    Decimal("0.00"))
    if ap_total != invoice[0]:
        return f"journal group {jrnl_no} has {ap_total} on the AP account, " \
               f"but the invoice is for {invoice[0]}"
    return _imbalance(jrnl_no, net)


def _closed_period_posting(conn: psycopg.Connection, correction: Correction) -> str | None:
    jrnl_no = correction.corrects_jrnl_no
    problem = _left_open(jrnl_no, group_net(conn, jrnl_no))
    if problem:
        return problem
    closed = conn.execute(
        'SELECT DISTINCT h."PRD_ID" FROM "GLJRNH" h '
        'JOIN "GLPERD" p ON p."PRD_ID" = h."PRD_ID" '
        'WHERE h."REF_JRNL_NO" = %s AND p."STAT_CD" <> %s ORDER BY 1',
        (jrnl_no, "O")).fetchone()
    return f"a correction line posts to period {closed[0]}, which is not open" if closed else None


def _invalid_account(conn: psycopg.Connection, correction: Correction) -> str | None:
    jrnl_no = correction.corrects_jrnl_no
    net = group_net(conn, jrnl_no)
    active = {r[0] for r in conn.execute(
        'SELECT "ACCT_CD" FROM "GLCOA" WHERE "ACTV_FLG" = %s', ("Y",))}
    invalid = sorted(set(net) - active)
    if not invalid:
        return f"journal group {jrnl_no} has no missing or inactive account"
    for acct in invalid:
        if net[acct]:
            return f"account {acct.rstrip()} still carries {net[acct]} " \
                   f"in journal group {jrnl_no}"
    return _imbalance(jrnl_no, net)


_RULES: dict[FaultType, Callable[[psycopg.Connection, Correction], str | None]] = {
    FaultType.UNBALANCED_JOURNAL: _unbalanced_journal,
    FaultType.DUPLICATE_INVOICE: _duplicate_invoice,
    FaultType.AP_GL_MISMATCH: _ap_gl_mismatch,
    FaultType.CLOSED_PERIOD_POSTING: _closed_period_posting,
    FaultType.INVALID_ACCOUNT: _invalid_account,
}


def check_resolved(conn: psycopg.Connection, finding: Finding,
                   correction: Correction) -> str | None:
    if finding.type == "other":
        return "a finding of type other cannot be resolved"
    if correction.corrects_jrnl_no is None and finding.type != FaultType.DUPLICATE_INVOICE:
        return "correction names no journal to correct"
    return _RULES[finding.type](conn, correction)
