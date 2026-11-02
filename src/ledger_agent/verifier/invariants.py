"""Ledger-wide checks, read from the rows as they stand.

A journal group is a journal plus every journal whose REF_JRNL_NO points to it,
identified here by that first journal's number. Faults the data already holds
appear as violations too, so verify compares the set before a trial with the set
after it and fails only on what is new.

Violation kinds match the FaultType values, with one addition for payments.
"""

from decimal import Decimal
from typing import NamedTuple

import psycopg

from ledger_agent.models import FaultType

AP_ACCOUNT = "2000"
PAYMENT_EXCEEDS_INVOICE = "payment_exceeds_invoice"

# Absolute, so a payment one cent over its invoice is rounding and not a violation.
PAYMENT_TOLERANCE = Decimal("0.01")


class Violation(NamedTuple):
    kind: str
    ref: str


# Debit minus credit, per journal group and account.
_GROUP_NETS = """
    SELECT coalesce(h."REF_JRNL_NO", h."JRNL_NO") AS grp, l."ACCT_CD" AS acct,
           sum(l."DR_AMT" - l."CR_AMT") AS net
    FROM "GLJRNH" h JOIN "GLJRNL" l ON l."JRNL_NO" = h."JRNL_NO"
    GROUP BY 1, 2
"""

_UNBALANCED = f"""
    SELECT grp::text FROM ({_GROUP_NETS}) n GROUP BY grp HAVING sum(net) <> 0
"""

_DUPLICATES = """
    SELECT rtrim("VEND_CD") || ':' || trim("INV_NO") FROM "APINVH"
    WHERE "STAT_CD" <> 'V'
    GROUP BY "VEND_CD", trim("INV_NO") HAVING count(*) > 1
"""

# Signed, so a credit memo's negative INV_AMT ties to its debit to AP. A void
# invoice should leave nothing on the AP account.
_AP_MISMATCHES = f"""
    SELECT i."INV_ID"::text
    FROM "APINVH" i
    LEFT JOIN (SELECT grp, -sum(net) AS ap FROM ({_GROUP_NETS}) n
               WHERE rtrim(acct) = %(ap_account)s GROUP BY grp) a ON a.grp = i."JRNL_NO"
    WHERE coalesce(a.ap, 0) <> CASE WHEN i."STAT_CD" = 'V' THEN 0 ELSE i."INV_AMT" END
"""

_CLOSED_POSTINGS = f"""
    SELECT DISTINCT coalesce(h."REF_JRNL_NO", h."JRNL_NO")::text
    FROM "GLJRNH" h JOIN "GLPERD" p ON p."PRD_ID" = h."PRD_ID"
    WHERE h."POST_DT" > p."CLOSE_DT"
      AND coalesce(h."REF_JRNL_NO", h."JRNL_NO") IN (
          SELECT grp FROM ({_GROUP_NETS}) n WHERE net <> 0)
"""

_INVALID_ACCOUNTS = f"""
    SELECT n.grp::text || ':' || rtrim(n.acct)
    FROM ({_GROUP_NETS}) n LEFT JOIN "GLCOA" c ON c."ACCT_CD" = n.acct
    WHERE n.net <> 0 AND c."ACTV_FLG" IS DISTINCT FROM 'Y'
"""

# A void invoice is owed nothing, so a payment against one is always over.
_OVERPAYMENTS = """
    SELECT i."INV_ID"::text
    FROM "APINVH" i JOIN "APPAYM" p ON p."INV_ID" = i."INV_ID"
    GROUP BY i."INV_ID", i."INV_AMT", i."STAT_CD"
    HAVING sum(p."PAY_AMT") >
           CASE WHEN i."STAT_CD" = 'V' THEN 0 ELSE i."INV_AMT" END + %(tolerance)s
"""

_CHECKS = (
    (FaultType.UNBALANCED_JOURNAL.value, _UNBALANCED),
    (FaultType.DUPLICATE_INVOICE.value, _DUPLICATES),
    (FaultType.AP_GL_MISMATCH.value, _AP_MISMATCHES),
    (FaultType.CLOSED_PERIOD_POSTING.value, _CLOSED_POSTINGS),
    (FaultType.INVALID_ACCOUNT.value, _INVALID_ACCOUNTS),
    (PAYMENT_EXCEEDS_INVOICE, _OVERPAYMENTS),
)


def group_net(conn: psycopg.Connection, jrnl_no: int) -> dict[str, Decimal]:
    """Debit minus credit by stored account code, for the group led by jrnl_no."""
    return dict(conn.execute(
        'SELECT acct, net FROM (' + _GROUP_NETS + ') n WHERE grp = %s', (jrnl_no,)).fetchall())


def global_violations(conn: psycopg.Connection) -> set[Violation]:
    params = {"ap_account": AP_ACCOUNT, "tolerance": PAYMENT_TOLERANCE}
    found = set()
    for kind, query in _CHECKS:
        rows = conn.execute(query, params if "%(" in query else None).fetchall()
        found |= {Violation(kind, ref) for (ref,) in rows}
    return found
