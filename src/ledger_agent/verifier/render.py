"""Turn a Correction into parameterised SQL, and into text for people to read.

Each Statement writes exactly one row. Rendering is the only source of the SQL
the verifier runs, and it writes GLJRNH, GLJRNL and APINVH.STAT_CD, nothing else.
"""

from decimal import Decimal
from typing import NamedTuple

from ledger_agent.models import Correction

POST_FLG = "P"
SRC_CD = "GL"
SRC_REF = "ledger_agent"

# POST_DT comes from GLPERD, so a correction posts on the first day of its period.
INSERT_HEADER = (
    'INSERT INTO "GLJRNH" ("JRNL_NO", "PRD_ID", "POST_DT", "POST_FLG", "REF_JRNL_NO", '
    '"SRC_CD", "SRC_REF", "MEMO") '
    'SELECT %s::integer, "PRD_ID", "STRT_DT", %s::char(1), %s::integer, %s::char(2), '
    '%s::text, %s::text FROM "GLPERD" WHERE "PRD_ID" = %s'
)
INSERT_LINE = (
    'INSERT INTO "GLJRNL" ("JRNL_NO", "LN_SEQ", "ACCT_CD", "DR_AMT", "CR_AMT") '
    'VALUES (%s, %s, %s, %s, %s)'
)
UPDATE_STATUS = 'UPDATE "APINVH" SET "STAT_CD" = %s WHERE "INV_ID" = %s'


class Statement(NamedTuple):
    sql: str
    params: tuple


def render(correction: Correction, next_jrnl_no: int) -> list[Statement]:
    """Render header, lines, then status changes. No lines means no journal."""
    statements = []
    if correction.lines:
        memo = f"Correction of journal {correction.corrects_jrnl_no}"
        statements.append(Statement(INSERT_HEADER, (
            next_jrnl_no, POST_FLG, correction.corrects_jrnl_no, SRC_CD, SRC_REF, memo,
            correction.lines[0].prd_id)))
    for seq, line in enumerate(correction.lines, start=1):
        statements.append(Statement(INSERT_LINE, (
            next_jrnl_no, seq, line.acct_cd, line.dr_amt, line.cr_amt)))
    for change in correction.status_changes:
        statements.append(Statement(UPDATE_STATUS, (change.new_status, change.inv_id)))
    return statements


def _literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, Decimal):
        return format(value, "f")
    return str(value)


def _inline(statement: Statement) -> str:
    parts = statement.sql.split("%s")
    values = [_literal(v) for v in statement.params] + [""]
    return "".join(part + value for part, value in zip(parts, values, strict=True))


def render_text(correction: Correction, next_jrnl_no: int) -> str:
    """The rendered statements with literal values. For reading, never for running."""
    return "\n".join(_inline(s) + ";" for s in render(correction, next_jrnl_no))
