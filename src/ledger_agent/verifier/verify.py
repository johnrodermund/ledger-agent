"""Decide whether a structured correction is acceptable.

verify tries the correction in a transaction that always rolls back: the
connection is in autocommit, so the transaction is opened by hand, and no code
here issues COMMIT. Checks run in order and the first failure is the result.
Reasons go back to the LLM and into the replay hash, so they carry only values
from the correction and the data.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal

import psycopg

from ledger_agent.db import connect_rw
from ledger_agent.models import Correction, CorrectionLine, Finding
from ledger_agent.verifier.invariants import Violation, global_violations, group_net
from ledger_agent.verifier.render import Statement, render
from ledger_agent.verifier.rules import check_resolved

MAX_CORRECTION_ROWS = 20
TRIAL_TIMEOUT_MS = 5000

# The snapshot of each table excludes only what a correction may change: its new
# journal, and the status of the invoices it names.
_FROZEN = {t: f'SELECT * FROM "{t}"'
           for t in ("GLPERD", "GLCOA", "APVEND", "APPAYM", "SYSDICT")}
_NOT_NEW_JOURNAL = {t: f'SELECT * FROM "{t}" WHERE "JRNL_NO" <> %(jrnl_no)s'
                    for t in ("GLJRNH", "GLJRNL")}
_INVOICES_WITHOUT_STATUS = (
    'SELECT "INV_ID", "VEND_CD", "INV_NO", "INV_DT", "INV_AMT", "JRNL_NO", '
    'CASE WHEN "INV_ID" = ANY(%(inv_ids)s) THEN NULL ELSE "STAT_CD" END AS "STAT_CD" '
    'FROM "APINVH"'
)
_SNAPSHOTS = {**_FROZEN, **_NOT_NEW_JOURNAL, "APINVH": _INVOICES_WITHOUT_STATUS}


@dataclass(frozen=True)
class VerifyResult:
    passed: bool
    check: str
    reason: str


@dataclass
class _Trial:
    """What the checks share. _trial_run and _resolution fill in the defaults."""

    conn: psycopg.Connection
    finding: Finding
    correction: Correction
    next_jrnl_no: int = 0
    statements: list[Statement] = field(default_factory=list)
    before: dict[str, tuple] = field(default_factory=dict)
    violations_before: set[Violation] = field(default_factory=set)
    violations_after: set[Violation] = field(default_factory=set)


def _database_failure(e: psycopg.Error) -> VerifyResult:
    if isinstance(e, psycopg.errors.QueryCanceled):
        return VerifyResult(False, "trial_run",
                            f"trial run exceeded the {TRIAL_TIMEOUT_MS} ms time limit")
    # Primary message only: no connection details, positions or hints.
    message = e.diag.message_primary or type(e).__name__
    return VerifyResult(False, "trial_run", f"database error: {message}")


def _bad_account_problem(lines: list[CorrectionLine], active: dict[str, str],
                         net: dict[str, Decimal], original: int | None) -> str | None:
    """A missing or inactive account may only be worked down towards zero: on the
    side that reduces its net in the group, and by no more than that net."""
    moved: dict[str, Decimal] = defaultdict(Decimal)
    for line in lines:
        acct = line.acct_cd
        if active.get(acct) == "Y":
            continue
        name = acct.rstrip()
        state = "is inactive" if acct in active else "does not exist"
        held = net.get(acct, 0)
        if not held:
            return f"account {name} {state} and carries no balance on journal {original}"
        reduced = line.cr_amt if held > 0 else line.dr_amt
        if not reduced:
            return f"account {name} {state} and may only be reduced: "                    f"journal {original} nets {held} on it"
        moved[acct] += reduced
        if moved[acct] > abs(held):
            return f"account {name} {state} and correction moves {moved[acct]} on it, "                    f"more than the {held} net on journal {original}"
    return None


def _structure_problem(trial: _Trial) -> str | None:
    conn, correction = trial.conn, trial.correction
    lines, changes = correction.lines, correction.status_changes
    original = correction.corrects_jrnl_no

    if trial.finding.type == "other":
        return "a finding of type other cannot be corrected"
    if not lines and not changes:
        return "correction changes nothing: it has no lines and no status changes"
    if original is None and not changes:
        return "correction names no journal to correct and no status changes"
    if original is None and lines:
        return "correction has lines but names no journal to correct"
    if len({ln.prd_id for ln in lines}) > 1:
        return "all lines must post to one period, as they share one journal header"
    seen: set[int] = set()
    for change in changes:
        if change.inv_id in seen:
            return f"invoice {change.inv_id} has more than one status change"
        seen.add(change.inv_id)

    net: dict[str, Decimal] = {}
    if original is not None:
        header = conn.execute('SELECT "REF_JRNL_NO" FROM "GLJRNH" WHERE "JRNL_NO" = %s',
                              (original,)).fetchone()
        if header is None:
            return f"journal {original} does not exist"
        if header[0] is not None:
            return f"journal {original} is itself a correction of journal {header[0]}"
        net = group_net(conn, original)

    active = dict(conn.execute('SELECT "ACCT_CD", "ACTV_FLG" FROM "GLCOA"').fetchall())
    problem = _bad_account_problem(lines, active, net, original)
    if problem:
        return problem

    if lines:
        prd_id = lines[0].prd_id
        period = conn.execute('SELECT "STAT_CD" FROM "GLPERD" WHERE "PRD_ID" = %s',
                              (prd_id,)).fetchone()
        if period is None:
            return f"period {prd_id} does not exist"
        if period[0] != "O":
            return f"period {prd_id} is not open"

    for change in changes:
        found = conn.execute('SELECT "STAT_CD" FROM "APINVH" WHERE "INV_ID" = %s',
                             (change.inv_id,)).fetchone()
        if found is None:
            return f"invoice {change.inv_id} does not exist"
        if found[0] == "V":
            return f"invoice {change.inv_id} is already void"
    return None


def _structure(trial: _Trial) -> VerifyResult | None:
    problem = _structure_problem(trial)
    return VerifyResult(False, "structure", problem) if problem else None


def _snapshot(trial: _Trial) -> dict[str, tuple]:
    params = {"jrnl_no": trial.next_jrnl_no,
              "inv_ids": [c.inv_id for c in trial.correction.status_changes]}
    return {table: trial.conn.execute(
        "SELECT count(*), md5(coalesce(string_agg(t::text, E'\\n' ORDER BY t::text), '')) "
        f"FROM ({query}) AS t", params if "%(" in query else None).fetchone()
            for table, query in _SNAPSHOTS.items()}


def _trial_run(trial: _Trial) -> VerifyResult | None:
    conn = trial.conn
    try:
        conn.execute(f"SET LOCAL statement_timeout = {int(TRIAL_TIMEOUT_MS)}")
        trial.next_jrnl_no = conn.execute(
            'SELECT coalesce(max("JRNL_NO"), 0) + 1 FROM "GLJRNH"').fetchone()[0]
        trial.before = _snapshot(trial)
        trial.violations_before = global_violations(conn)
        trial.statements = render(trial.correction, trial.next_jrnl_no)
        for statement in trial.statements:
            conn.execute(statement.sql, statement.params)
    except psycopg.Error as e:
        return _database_failure(e)
    return None


def _change_limits(trial: _Trial) -> VerifyResult | None:
    def failed(reason: str) -> VerifyResult:
        return VerifyResult(False, "change_limits", reason)

    if len(trial.statements) > MAX_CORRECTION_ROWS:
        return failed(f"correction writes {len(trial.statements)} rows, "
                      f"over the limit of {MAX_CORRECTION_ROWS}")
    try:
        after = _snapshot(trial)
        conn, jrnl_no, correction = trial.conn, trial.next_jrnl_no, trial.correction
        new_headers = conn.execute('SELECT count(*) FROM "GLJRNH" WHERE "JRNL_NO" = %s',
                                   (jrnl_no,)).fetchone()[0]
        new_lines = conn.execute('SELECT count(*) FROM "GLJRNL" WHERE "JRNL_NO" = %s',
                                 (jrnl_no,)).fetchone()[0]
        voided = conn.execute(
            'SELECT count(*) FROM "APINVH" WHERE "INV_ID" = ANY(%s) AND "STAT_CD" = %s',
            ([c.inv_id for c in correction.status_changes], "V")).fetchone()[0]
    except psycopg.Error as e:
        return _database_failure(e)

    for table in _SNAPSHOTS:
        if after[table] != trial.before[table]:
            return failed(f"table {table} changed beyond the rendered rows")
    if new_headers != (1 if correction.lines else 0):
        return failed(f"expected {1 if correction.lines else 0} new GLJRNH rows, "
                      f"found {new_headers}")
    if new_lines != len(correction.lines):
        return failed(f"expected {len(correction.lines)} new GLJRNL rows, found {new_lines}")
    if voided != len(correction.status_changes):
        return failed(f"expected {len(correction.status_changes)} APINVH status changes, "
                      f"found {voided}")
    return None


def _resolution(trial: _Trial) -> VerifyResult | None:
    """The finding's own rule, then proof the correction cleared a violation of that
    kind, so a correction of some healthy journal cannot pass as a fix."""
    try:
        problem = check_resolved(trial.conn, trial.finding, trial.correction)
        trial.violations_after = global_violations(trial.conn)
    except psycopg.Error as e:
        return _database_failure(e)
    if problem is None:
        cleared = trial.violations_before - trial.violations_after
        if not any(v.kind == trial.finding.type for v in cleared):
            problem = f"correction clears no {trial.finding.type} violation"
    return VerifyResult(False, "resolution", problem) if problem else None


def _collateral(trial: _Trial) -> VerifyResult | None:
    new = sorted(trial.violations_after - trial.violations_before)
    if not new:
        return None
    listed = ", ".join(f"{v.kind} {v.ref}" for v in new)
    return VerifyResult(False, "collateral", f"correction introduces {listed}")


_CHECKS: tuple[Callable[[_Trial], VerifyResult | None], ...] = (
    _structure, _trial_run, _change_limits, _resolution, _collateral)


def _rollback_and_close(conn: psycopg.Connection) -> None:
    try:
        conn.execute("ROLLBACK")
    except psycopg.Error:
        pass  # The connection is gone, and its open transaction went with it.
    finally:
        conn.close()


def verify(dbname: str, finding: Finding, correction: Correction) -> VerifyResult:
    conn = connect_rw(dbname)
    try:
        conn.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
        trial = _Trial(conn, finding, correction)
        for check in _CHECKS:
            failure = check(trial)
            if failure:
                return failure
        return VerifyResult(True, "ok", "correction passed all checks")
    finally:
        _rollback_and_close(conn)
