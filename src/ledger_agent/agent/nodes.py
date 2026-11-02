"""Graph nodes for discovery: explore the schema, hunt for findings, confirm them."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ledger_agent.agent.loop import LoopResult, format_query_result, run_tool_loop
from ledger_agent.agent.prompts import (
    EXPLORE_START,
    EXPLORE_SYSTEM,
    FINDING_TYPES,
    HUNT_START,
    HUNT_SYSTEM,
    PROPOSE_PROMPT,
    PROPOSE_RETRY,
    PROPOSE_SYSTEM,
    SUBMIT_CORRECTION_TOOL,
    SUBMIT_FINDINGS_TOOL,
    SUBMIT_SCHEMA_NOTES_TOOL,
)
from ledger_agent.agent.state import AgentState, Deps, DroppedFinding, Evidence, Outcome
from ledger_agent.db import connect_ro
from ledger_agent.llm.client import (
    DEFAULT_MODEL,
    ModelClient,
    ModelRequest,
    assistant_message,
    tool_result_message,
)
from ledger_agent.models import Correction, Finding, RowRef
from ledger_agent.tools.sql import QueryResult, run_query
from ledger_agent.verifier.render import render_text
from ledger_agent.verifier.verify import verify as verify_correction

__all__ = ["Deps", "explore", "hunt", "confirm", "pick", "propose", "verify"]

EVIDENCE_ROWS = 20
TOKEN_LIMIT_REASON = "token limit reached"


def _record_limits(hit: list[str], stage: str, result: LoopResult) -> list[str]:
    hit = list(hit)
    if result.limit_kind == "queries":
        hit.append(f"{stage}_queries")
    elif result.limit_kind == "tokens" and "tokens" not in hit:
        hit.append("tokens")
    if result.failure:
        hit.append(f"{stage}_failed: {result.failure}")
    return hit


def explore(state: AgentState, deps: Deps) -> dict[str, Any]:
    result = run_tool_loop(
        deps.client,
        EXPLORE_SYSTEM,
        [{"role": "user", "content": EXPLORE_START}],
        deps.limits.explore_queries,
        SUBMIT_SCHEMA_NOTES_TOOL,
        state["dbname"],
        token_limit=deps.limits.tokens,
    )
    notes = result.submitted.get("notes", "")
    return {
        "schema_notes": notes if isinstance(notes, str) else "",
        "limits_hit": _record_limits(state["limits_hit"], "explore", result),
        "queries": state.get("queries", 0) + result.queries_used,
    }


def _parse_key(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("key must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise ValueError("key must be an integer")


def _parse_finding(finding_id: str, raw: object) -> Finding:
    """Build a Finding from model output; raises ValueError describing the problem."""
    if not isinstance(raw, dict):
        raise ValueError("finding must be an object")
    type_ = raw.get("type")
    if type_ not in FINDING_TYPES:
        raise ValueError(f"unknown finding type {type_!r}")
    summary, sql = raw.get("summary"), raw.get("evidence_sql")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("summary is required")
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("evidence_sql is required")
    rows_raw = raw.get("affected_rows")
    if not isinstance(rows_raw, list):
        raise ValueError("affected_rows must be a list")
    rows: list[RowRef] = []
    for item in rows_raw:
        if not isinstance(item, dict) or not isinstance(item.get("table"), str) \
                or not item["table"].strip():
            raise ValueError("each affected row needs a table name")
        ref = RowRef(table=item["table"].strip().upper(), key=_parse_key(item.get("key")))
        if ref not in rows:
            rows.append(ref)
    return Finding(
        finding_id=finding_id, type=type_, summary=summary, affected_rows=rows, evidence_sql=sql
    )


def hunt(state: AgentState, deps: Deps) -> dict[str, Any]:
    notes = state["schema_notes"] or "(no schema notes were produced)"
    result = run_tool_loop(
        deps.client,
        HUNT_SYSTEM,
        [{"role": "user", "content": HUNT_START.format(notes=notes)}],
        deps.limits.hunt_queries,
        SUBMIT_FINDINGS_TOOL,
        state["dbname"],
        token_limit=deps.limits.tokens,
    )
    submitted = result.submitted.get("findings", [])
    if not isinstance(submitted, list):
        submitted = []
    findings: list[Finding] = []
    dropped = list(state["dropped"])
    for index, raw in enumerate(submitted, start=1):
        finding_id = f"F{index:03d}"
        try:
            findings.append(_parse_finding(finding_id, raw))
        except ValueError as e:  # pydantic's ValidationError is a ValueError
            summary = raw.get("summary") if isinstance(raw, dict) else None
            dropped.append(DroppedFinding(
                finding_id=finding_id,
                summary=summary if isinstance(summary, str) else "",
                reason=f"invalid submission: {e}",
            ))
    return {
        "findings": findings,
        "dropped": dropped,
        "limits_hit": _record_limits(state["limits_hit"], "hunt", result),
        "queries": state.get("queries", 0) + result.queries_used,
    }


def _merge(findings: list[Finding]) -> list[Finding]:
    """Merge findings of the same type that share a row. The earliest id wins."""
    merged: list[Finding] = []
    for f in findings:
        rows = list(f.affected_rows)
        keep = f
        absorbed: list[Finding] = []
        for m in merged:
            if m.type == f.type and set(m.affected_rows) & set(rows):
                absorbed.append(m)
        for m in absorbed:
            merged.remove(m)
            rows = m.affected_rows + [r for r in rows if r not in m.affected_rows]
            if m.finding_id < keep.finding_id:
                keep = m
        merged.append(keep.model_copy(update={"affected_rows": rows}))
    return sorted(merged, key=lambda x: x.finding_id)


def _evidence(result: QueryResult) -> Evidence:
    return Evidence(columns=result.columns, rows=result.rows[:EVIDENCE_ROWS],
                    truncated=result.truncated or len(result.rows) > EVIDENCE_ROWS)


def confirm(state: AgentState, deps: Deps) -> dict[str, Any]:
    dropped = list(state["dropped"])
    kept: list[Finding] = []
    evidence: dict[str, Evidence] = {}
    for f in state["findings"]:
        result = run_query(state["dbname"], f.evidence_sql)
        if not result.ok:
            reason = f"evidence query rejected or failed: {result.error}"
        elif not result.rows:
            reason = "evidence query returned no rows"
        else:
            kept.append(f)
            evidence[f.finding_id] = _evidence(result)
            continue
        dropped.append(DroppedFinding(finding_id=f.finding_id, summary=f.summary, reason=reason))
    merged = _merge(kept)
    return {
        "findings": merged,
        "dropped": dropped,
        "evidence": {f.finding_id: evidence[f.finding_id] for f in merged},
        "queue": [f.finding_id for f in merged],
        "queries": state.get("queries", 0) + len(state["findings"]),
    }


# Propose and verify ---------------------------------------------------------------------


class _TokenLimit(Exception):
    pass


def _validation_message(e: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'correction'}: {err['msg']}"
        for err in e.errors()
    )


def _parse_correction(resp: Any) -> tuple[Correction | None, str | None]:
    """The submitted correction, or a description of what was wrong with the reply."""
    call = next((c for c in resp.tool_calls if c.name == SUBMIT_CORRECTION_TOOL["name"]), None)
    if call is None:
        return None, "No submission was made."
    try:
        return Correction.model_validate(call.arguments), None
    except ValidationError as e:
        return None, f"The submission is invalid: {_validation_message(e)}."


def _request_correction(
    client: ModelClient, system: str, prompt: str, token_limit: int,
    malformed: list[dict[str, Any]],
) -> tuple[Correction | None, str | None]:
    """One model call, plus one retry carrying the validation error if the reply is
    malformed. A second failure returns (None, error). Raises _TokenLimit when the
    budget runs out before the retry. This is not a fix attempt. Submissions that fail
    validation are added to malformed, so they stay on record however this ends."""
    tools = [SUBMIT_CORRECTION_TOOL]
    msgs: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    problem = ""
    for retry in (False, True):
        if retry and client.tokens_used >= token_limit:
            raise _TokenLimit
        resp = client.complete(ModelRequest(DEFAULT_MODEL, system, msgs, tools))
        correction, error = _parse_correction(resp)
        if correction is not None:
            return correction, None
        problem = error or ""
        malformed.extend(c.arguments for c in resp.tool_calls
                         if c.name == SUBMIT_CORRECTION_TOOL["name"])
        note = {"type": "text", "text": f"{problem} Respond by calling the "
                f"`{SUBMIT_CORRECTION_TOOL['name']}` tool with a valid correction."}
        assistant = assistant_message(resp)
        if assistant["content"]:
            msgs.append(assistant)
        if resp.tool_calls:
            feedback = tool_result_message([(c.id, problem, True) for c in resp.tool_calls])
            feedback["content"].append(note)
            msgs.append(feedback)
        else:
            msgs.append({"role": "user", "content": [note]})
    return None, problem


def _find(state: AgentState) -> Finding:
    return next(f for f in state["findings"] if f.finding_id == state["current"])


def _end_finding(state: AgentState, outcome: Outcome) -> dict[str, Any]:
    return {
        "outcomes": [*state["outcomes"], outcome],
        "current": None, "attempt": 0, "last_failure": None, "proposal": None,
    }


def _stop_for_tokens(state: AgentState) -> dict[str, Any]:
    """Out of tokens: the current finding and everything still queued need a person."""
    pending = ([state["current"]] if state["current"] else []) + list(state["queue"])
    outcomes = [*state["outcomes"], *(
        Outcome(finding_id=fid, status="needs_human",
                attempts=state["attempt"] if fid == state["current"] else 0,
                reason=TOKEN_LIMIT_REASON) for fid in pending)]
    hit = list(state["limits_hit"])
    if "tokens" not in hit:
        hit.append("tokens")
    return {"outcomes": outcomes, "queue": [], "current": None, "attempt": 0,
            "last_failure": None, "proposal": None, "limits_hit": hit}


def pick(state: AgentState, deps: Deps) -> dict[str, Any]:
    queue = list(state["queue"])
    if not queue:
        return {"current": None}
    return {"current": queue[0], "queue": queue[1:], "attempt": 0, "last_failure": None,
            "proposal": None}


def _propose_prompt(state: AgentState, finding: Finding) -> str:
    evidence = state["evidence"].get(finding.finding_id)
    if evidence is None:
        shown = "(none recorded)"
    else:
        shown = format_query_result(QueryResult(
            ok=True, columns=evidence.columns, rows=evidence.rows, truncated=evidence.truncated))
    rows = ", ".join(f"{r.table} {r.key}" for r in finding.affected_rows) or "(none)"
    retry = ""
    if state["last_failure"] and state["proposal"] is not None:
        retry = PROPOSE_RETRY.format(
            previous=state["attempt"], reason=state["last_failure"],
            correction=state["proposal"].model_dump_json())
    return PROPOSE_PROMPT.format(
        notes=state["schema_notes"] or "(no schema notes were produced)",
        finding_id=finding.finding_id, finding_type=finding.type, summary=finding.summary,
        rows=rows, evidence_sql=finding.evidence_sql, evidence=shown, retry=retry)


def propose(state: AgentState, deps: Deps) -> dict[str, Any]:
    finding = _find(state)
    if finding.type == "other":
        return _end_finding(state, Outcome(
            finding_id=finding.finding_id, status="needs_human", attempts=0,
            reason="no correction is proposed for a finding of type other"))
    if deps.client.tokens_used >= deps.limits.tokens:
        return _stop_for_tokens(state)
    malformed: list[dict[str, Any]] = []
    try:
        correction, problem = _request_correction(
            deps.client, PROPOSE_SYSTEM, _propose_prompt(state, finding), deps.limits.tokens,
            malformed)
    except _TokenLimit:
        return _stop_for_tokens(state) | {"malformed": [*state["malformed"], *malformed]}
    kept = {"malformed": [*state["malformed"], *malformed]}
    if correction is None:
        return _end_finding(state, Outcome(
            finding_id=finding.finding_id, status="needs_human", attempts=state["attempt"],
            reason=f"the model did not produce a valid correction: {problem}")) | kept
    return {"proposal": correction} | kept


def _next_jrnl_no(dbname: str) -> int:
    with connect_ro(dbname) as conn:
        return conn.execute('SELECT coalesce(max("JRNL_NO"), 0) + 1 FROM "GLJRNH"').fetchone()[0]


def verify(state: AgentState, deps: Deps) -> dict[str, Any]:
    finding, correction = _find(state), state["proposal"]
    if correction is None:
        raise RuntimeError("verify ran without a proposal")
    attempt = state["attempt"] + 1
    result = verify_correction(state["dbname"], finding, correction)
    if result.passed:
        return _end_finding(state, Outcome(
            finding_id=finding.finding_id, status="verified", correction=correction,
            attempts=attempt,
            rendered_sql=render_text(correction, _next_jrnl_no(state["dbname"]))))
    failure = f"{result.check}: {result.reason}"
    rejected = [*state["rejected"], correction]
    if attempt >= deps.limits.fix_attempts:
        return _end_finding(state, Outcome(
            finding_id=finding.finding_id, status="needs_human", attempts=attempt,
            reason=f"verifier rejected every correction; last: {failure}")) | {"rejected": rejected}
    return {"attempt": attempt, "last_failure": failure, "rejected": rejected}
