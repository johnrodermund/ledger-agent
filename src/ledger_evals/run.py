"""Run the agent across the eval scenarios and score it against each answer key.

python -m ledger_evals.run (--split dev|holdout|injection|all | --scenario ID) --out DIR

Each scenario loads into its own database and records under its own namespace, with
a fresh client, because the token budget compares against a client's lifetime total.
A scenario that fails aborts the run and nothing is written: a failure is never
scored. Thresholds are checked against the headline aggregate, which is holdout when
all of holdout ran. When thresholds are set, a metric with no value and an incomplete
scenario each count as missed. A failed injection check exits non-zero regardless.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ledger_agent.agent.graph import audit
from ledger_agent.agent.state import AuditResult, Deps
from ledger_agent.erp.faults import build_scenario
from ledger_agent.erp.load import load
from ledger_agent.llm.client import DEFAULT_MODEL, client_from_env
from ledger_agent.models import AnswerKey, ScenarioSpec
from ledger_evals.scenarios import SCENARIOS_PATH, load_scenarios
from ledger_evals.score import ScenarioScore, Tally, aggregate, score

THRESHOLDS_PATH = SCENARIOS_PATH.with_name("thresholds.json")
HIGHER_IS_BETTER = ("recall", "precision", "fix_validity", "end_to_end_fix_rate")
LOWER_IS_BETTER = ("decoy_false_alarm_rate", "needs_human_rate")
SPLITS = ("dev", "holdout", "injection")


@dataclass
class ScenarioRun:
    spec: ScenarioSpec
    key: AnswerKey
    result: AuditResult
    payment_invoices: dict[int, int]
    llm_mode: str


@dataclass
class Summary:
    label: str
    headline: Tally
    aggregates: dict[str, Tally]


def db_name(scenario_id: str) -> str:
    return f"ledger_{scenario_id.replace('-', '_')}"


def run_scenario(spec: ScenarioSpec) -> ScenarioRun:
    dataset, key = build_scenario(spec)
    db = db_name(spec.id)
    load(db, dataset)
    client = client_from_env(namespace=spec.id)
    payments = {p["PAY_ID"]: p["INV_ID"] for p in dataset.tables["APPAYM"]}
    return ScenarioRun(spec, key, audit(db, Deps(client)), payments, client.mode)


def plain_variant(spec: ScenarioSpec, specs: list[ScenarioSpec]) -> ScenarioSpec:
    for other in specs:
        if (not other.hostile_memo and (other.seed, other.faults, other.decoys)
                == (spec.seed, spec.faults, spec.decoys)):
            return other
    raise ValueError(f"scenario {spec.id} has no plain variant to compare recall against")


def select(specs: list[ScenarioSpec], split: str, scenario: str | None
           ) -> tuple[list[ScenarioSpec], list[ScenarioSpec]]:
    """The scenarios asked for, then every plain variant they need that was not asked
    for. The second list only supports the comparison and stays out of the aggregates."""
    if scenario is not None:
        requested = [s for s in specs if s.id == scenario]
        if not requested:
            raise ValueError(f"unknown scenario {scenario}")
    else:
        requested = [s for s in specs if split in ("all", s.split)]
    if not requested:
        raise ValueError(f"no scenarios in split {split}")
    extra: list[ScenarioSpec] = []
    for spec in requested:
        if spec.hostile_memo:
            plain = plain_variant(spec, specs)
            if plain not in requested and plain not in extra:
                extra.append(plain)
    return requested, extra


def _validate_thresholds(thresholds: dict[str, float]) -> None:
    unknown = set(thresholds) - set(HIGHER_IS_BETTER) - set(LOWER_IS_BETTER)
    if unknown:
        raise ValueError(f"unknown threshold metric: {', '.join(sorted(unknown))}")


def check_thresholds(total: Tally, thresholds: dict[str, float]) -> list[str]:
    """One message for each threshold missed."""
    _validate_thresholds(thresholds)
    missed = []
    if thresholds and total.incomplete_scenarios:
        missed.append(f"{total.incomplete_scenarios} scenario(s) incomplete")
    for metric, bound in thresholds.items():
        value = getattr(total, metric)
        lower = metric in LOWER_IS_BETTER
        if value is None:
            missed.append(f"{metric} has no value, needs {'at most' if lower else 'at least'} "
                          f"{bound}")
        elif value > bound if lower else value < bound:
            missed.append(f"{metric} is {value:.3f}, needs {'at most' if lower else 'at least'} "
                          f"{bound}")
    return missed


def _score_all(runs: list[ScenarioRun], specs: list[ScenarioSpec]) -> dict[str, ScenarioScore]:
    scores: dict[str, ScenarioScore] = {}
    for run in sorted(runs, key=lambda r: r.spec.hostile_memo):  # plain variants first
        plain = scores[plain_variant(run.spec, specs).id] if run.spec.hostile_memo else None
        scores[run.spec.id] = score(run.result, run.key, payment_invoices=run.payment_invoices,
                                    plain=plain)
    return scores


def summarise(scores: dict[str, ScenarioScore], requested: list[ScenarioSpec],
              specs: list[ScenarioSpec]) -> Summary:
    """Aggregates cover what was asked for, so a supporting plain variant adds nothing.
    Only the whole holdout split is the holdout aggregate."""
    aggregates = {name: aggregate(scores[s.id] for s in requested if s.split == name)
                  for name in SPLITS if any(s.split == name for s in requested)}
    holdout = [s for s in specs if s.split == "holdout"]
    if "holdout" in aggregates and all(s in requested for s in holdout):
        return Summary("Holdout aggregate", aggregates["holdout"], aggregates)
    names = ", ".join(aggregates)
    qualifier = "partial holdout" if "holdout" in aggregates else "not holdout"
    return Summary(f"Aggregate of {names} ({qualifier})",
                   aggregate(scores[s.id] for s in requested), aggregates)


def build_results(runs: list[ScenarioRun], scores: dict[str, ScenarioScore],
                  requested: list[ScenarioSpec], summary: Summary, thresholds: dict[str, float],
                  missed: list[str]) -> dict:
    return {
        "llm_mode": runs[0].llm_mode,
        "model": DEFAULT_MODEL,
        "headline": {"label": summary.label, **summary.headline.model_dump(mode="json")},
        "aggregates": {name: t.model_dump(mode="json") for name, t in summary.aggregates.items()},
        "scenarios": [
            {"id": run.spec.id, "split": run.spec.split, "requested": run.spec in requested,
             "score": scores[run.spec.id].model_dump(mode="json")}
            for run in runs],
        "thresholds": {"set": thresholds, "missed": missed},
    }


def _percent(part: int, whole: int) -> str:
    return f"{part / whole:.1%} ({part} of {whole})" if whole else "n/a"


def _metric_rows(t: Tally) -> list[str]:
    planted, detected = sum(t.faults_by_type.values()), sum(t.detected_by_type.values())
    return [
        "| Metric | Value |", "| --- | --- |",
        f"| Recall | {_percent(detected, planted)} |",
        f"| Precision | {_percent(t.matched_findings, t.findings)} |",
        f"| Decoy false alarm rate | {_percent(t.decoys_alarmed, t.decoys)} |",
        f"| Fix validity | {_percent(t.valid_fixes, t.verified_matched)} |",
        f"| Verified but wrong | {t.verified_but_wrong} |",
        f"| Verified fixes for findings credited to no fault | {t.verified_unmatched} |",
        f"| End-to-end fix rate | {_percent(t.valid_fixes, planted)} |",
        f"| Needs human rate | {_percent(t.needs_human, t.outcomes)} |",
        f"| Incomplete scenarios | {t.incomplete_scenarios} |",
        f"| Tokens | {t.tokens} |",
        f"| Cost (USD) | {t.cost_usd:.4f} |",
        f"| Queries | {t.queries} |",
    ]


def _recall_rows(t: Tally) -> list[str]:
    rows = ["| Fault type | Recall |", "| --- | --- |"]
    rows += [f"| {name} | {_percent(t.detected_by_type.get(name, 0), planted)} |"
             for name, planted in sorted(t.faults_by_type.items())]
    return rows


def _misses(run: ScenarioRun, one: ScenarioScore) -> list[str]:
    types = {f.fault_id: f.type.value for f in run.key.faults}
    lines = []
    if one.missed_faults:
        missed = ", ".join(f"{fid} ({types[fid]})" for fid in one.missed_faults)
        lines.append(f"- Missed faults: {missed}")
    if one.alarmed_decoys:
        lines.append(f"- Decoy false alarms: {', '.join(one.alarmed_decoys)}")
    if one.wrong_fixes:
        lines.append(f"- Verified but wrong: {', '.join(one.wrong_fixes)}")
    if one.unmatched_findings:
        lines.append(f"- Findings credited to no fault: {', '.join(one.unmatched_findings)}")
    if one.incomplete:
        lines.append(f"- Audit incomplete: {'; '.join(one.incomplete)}")
    return lines or ["- None."]


def render_markdown(runs: list[ScenarioRun], scores: dict[str, ScenarioScore],
                    requested: list[ScenarioSpec], summary: Summary, missed: list[str]) -> str:
    md = ["# LedgerAgent eval results", "",
          f"LLM mode: {runs[0].llm_mode}. Model: {DEFAULT_MODEL}.", "",
          f"## {summary.label}", "", *_metric_rows(summary.headline), "",
          "Each fault credits one finding at most, and a surplus finding counts as incorrect. "
          "Fix validity is over credited findings with a verified outcome. Verified but wrong "
          "counts fixes the verifier accepted that the answer key refuses. The end-to-end fix "
          "rate is planted faults with a valid fix, over all planted faults.", "",
          "## Recall by fault type", "", f"{summary.label}:", "",
          *_recall_rows(summary.headline), "",
          "## Misses by scenario", ""]
    for run in runs:
        note = "" if run.spec in requested else " (plain variant, not aggregated)"
        md += [f"### {run.spec.id}{note}", "", *_misses(run, scores[run.spec.id]), ""]

    injections = [run.spec.id for run in runs if run.spec.hostile_memo]
    if injections:
        md += ["## Injection", "", "| Scenario | Result |", "| --- | --- |"]
        md += [f"| {sid} | {'passed' if scores[sid].injection_passed else 'FAILED'} |"
               for sid in injections]
        md += [""]

    if len(summary.aggregates) > 1:
        for name, tally in summary.aggregates.items():
            if tally is not summary.headline:
                md += [f"## {name.capitalize()} split", "", *_metric_rows(tally), ""]

    md += ["## Thresholds", ""]
    md += [f"- Missed: {m}" for m in missed] or ["None missed."]
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ledger-evals")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--split", choices=[*SPLITS, "all"])
    which.add_argument("--scenario", help="run one scenario by id")
    parser.add_argument("--out", type=Path, required=True, help="results output directory")
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS_PATH)
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS_PATH)
    args = parser.parse_args(argv)

    try:
        specs = load_scenarios(args.scenarios)
        thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
        _validate_thresholds(thresholds)
        requested, extra = select(specs, args.split, args.scenario)
    except (OSError, ValueError) as e:
        print(f"error: {' '.join(str(e).split())}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("results.json", "results.md"):
        (args.out / name).unlink(missing_ok=True)  # an abort must not leave old results

    runs: list[ScenarioRun] = []
    for spec in sorted([*extra, *requested], key=lambda s: s.hostile_memo):
        print(f"Running scenario {spec.id}", flush=True)
        try:
            runs.append(run_scenario(spec))
        except Exception as e:  # whatever went wrong, this scenario has no valid result
            message = " ".join(str(e).split()) or type(e).__name__
            print(f"error: scenario {spec.id} failed, run aborted: {message}", file=sys.stderr)
            return 1

    scores = _score_all(runs, specs)
    summary = summarise(scores, requested, specs)
    missed = check_thresholds(summary.headline, thresholds)
    results = build_results(runs, scores, requested, summary, thresholds, missed)
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (args.out / "results.md").write_text(
        render_markdown(runs, scores, requested, summary, missed), encoding="utf-8")

    print(f"Results written to {args.out}")
    if not thresholds:
        print("No thresholds set")
    for message in missed:
        print(f"Threshold missed: {message}")
    failed_injections = [sid for sid, one in scores.items() if one.injection_passed is False]
    for sid in failed_injections:
        print(f"Injection check failed: {sid}")
    return 1 if missed or failed_injections else 0


if __name__ == "__main__":
    sys.exit(main())
