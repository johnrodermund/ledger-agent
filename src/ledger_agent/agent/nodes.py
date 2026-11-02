"""Graph nodes for discovery: explore the schema, hunt for findings, confirm them."""

from __future__ import annotations

from typing import Any

from ledger_agent.agent.loop import LoopResult, run_tool_loop
from ledger_agent.agent.prompts import (
    EXPLORE_START,
    EXPLORE_SYSTEM,
    FINDING_TYPES,
    HUNT_START,
    HUNT_SYSTEM,
    SUBMIT_FINDINGS_TOOL,
    SUBMIT_SCHEMA_NOTES_TOOL,
)
from ledger_agent.agent.state import AgentState, Deps, DroppedFinding
from ledger_agent.models import Finding, RowRef
from ledger_agent.tools.sql import run_query

__all__ = ["Deps", "explore", "hunt", "confirm"]


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


def confirm(state: AgentState, deps: Deps) -> dict[str, Any]:
    dropped = list(state["dropped"])
    kept: list[Finding] = []
    for f in state["findings"]:
        result = run_query(state["dbname"], f.evidence_sql)
        if not result.ok:
            reason = f"evidence query rejected or failed: {result.error}"
        elif not result.rows:
            reason = "evidence query returned no rows"
        else:
            kept.append(f)
            continue
        dropped.append(DroppedFinding(finding_id=f.finding_id, summary=f.summary, reason=reason))
    return {"findings": _merge(kept), "dropped": dropped}
