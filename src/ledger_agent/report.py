"""Write the audit result as report.md, report.json and trace.json.

Text that comes from the database or the model (memos, summaries, SQL, error
messages) goes inside code fences that are longer than any run of backticks in the
text, so it cannot change the structure of the Markdown around it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ledger_agent.agent.state import AuditResult, Evidence, Outcome

NOTHING_APPLIED = (
    "**Nothing was applied.** This audit only reads the database. Every correction below "
    "was checked in a transaction that was rolled back, and the SQL is shown for a person "
    "to review. The journal number in it is the next free number when the check ran."
)


def _fence(text: str, language: str = "") -> str:
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{language}\n{text.rstrip()}\n{ticks}"


def _evidence_text(evidence: Evidence | None) -> str:
    if evidence is None:
        return "(none recorded)"
    lines = [json.dumps(evidence.columns)]
    lines += [json.dumps(row) for row in evidence.rows]
    if evidence.truncated:
        lines.append("(more rows were returned than are shown)")
    return "\n".join(lines)


def _finding_section(result: AuditResult, outcome: Outcome) -> list[str]:
    state = result.state
    finding = next((f for f in state["findings"] if f.finding_id == outcome.finding_id), None)
    heading = f"### {outcome.finding_id}" + (f" ({finding.type})" if finding else "")
    out = [heading, ""]
    if finding is not None:
        rows = ", ".join(f"{r.table} {r.key}" for r in finding.affected_rows) or "none"
        out += [
            "Summary:", "", _fence(finding.summary), "",
            "Affected rows:", "", _fence(rows), "",
            "Evidence query:", "", _fence(finding.evidence_sql, "sql"), "",
            "Evidence rows:", "", _fence(_evidence_text(state["evidence"].get(outcome.finding_id))),
            "",
        ]
    if outcome.status == "verified":
        out += [f"Verified after {outcome.attempts} attempt(s). Proposed SQL, not applied:", ""]
        out += [_fence(outcome.rendered_sql or "", "sql"), ""]
        if outcome.correction is not None:
            out += ["Reasoning:", "", _fence(outcome.correction.reasoning), ""]
    else:
        out += [f"Needs a person after {outcome.attempts} fix attempt(s). Reason:", "",
                _fence(outcome.reason or "no reason recorded"), ""]
    return out


def render_markdown(result: AuditResult) -> str:
    state = result.state
    verified = [o for o in state["outcomes"] if o.status == "verified"]
    unresolved = [o for o in state["outcomes"] if o.status != "verified"]
    dropped = state["dropped"]
    md = [
        "# LedgerAgent audit report", "", NOTHING_APPLIED, "",
        "## Run summary", "",
        f"- Findings confirmed: {len(state['findings'])}",
        f"- Verified corrections: {len(verified)}",
        f"- Unresolved findings: {len(unresolved)}",
        f"- Dropped findings: {len(dropped)}",
        f"- Model tokens used: {result.tokens}",
        f"- Queries run: {result.queries}",
        "",
    ]
    if not state["findings"]:
        md += ["No discrepancies found.", ""]

    md += ["## Limits hit", ""]
    md += [_fence("\n".join(state["limits_hit"])), ""] if state["limits_hit"] else ["None.", ""]

    md += ["## Verified findings", ""]
    if verified:
        for outcome in verified:
            md += _finding_section(result, outcome)
    else:
        md += ["None.", ""]

    md += ["## Unresolved findings", ""]
    if unresolved:
        for outcome in unresolved:
            md += _finding_section(result, outcome)
    else:
        md += ["None.", ""]

    md += ["## Dropped findings", ""]
    if dropped:
        for d in dropped:
            md += [f"### {d.finding_id or 'unnumbered'}", "",
                   "Summary:", "", _fence(d.summary or "(none)"), "",
                   "Reason dropped:", "", _fence(d.reason), ""]
    else:
        md += ["None.", ""]
    return "\n".join(md)


def write_report(result: AuditResult, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(render_markdown(result), encoding="utf-8")
    (out_dir / "report.json").write_text(result.model_dump_json(indent=2) + "\n",
                                         encoding="utf-8")
    trace = [event.model_dump() for event in result.trace]
    (out_dir / "trace.json").write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
