"""Run the fix library through the verifier and report per check.

python -m ledger_evals.guardrails exits non-zero unless every bad fix is refused at
its expected check and every good fix passes. It loads each scenario into a scratch
database of its own and always drops it.
"""

import sys
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager

from ledger_agent.db import connect_admin
from ledger_agent.erp.faults import build_scenario
from ledger_agent.erp.generate import Dataset
from ledger_agent.erp.load import load
from ledger_agent.models import AnswerKey, DecoyType, FaultType, ScenarioSpec
from ledger_agent.verifier.verify import verify
from ledger_evals.bad_fixes import BAD_FIXES, GOOD_FIXES, FixCase

SEEDS = (7, 11, 23)
DB_PREFIX = "guardrails_"


def scenario_spec(seed: int) -> ScenarioSpec:
    return ScenarioSpec(id=f"guardrails-{seed}", split="dev", seed=seed,
                        faults={t: 1 for t in FaultType}, decoys=list(DecoyType))


@contextmanager
def _scratch_database(dataset: Dataset) -> Iterator[str]:
    name = f"{DB_PREFIX}{uuid.uuid4().hex[:12]}"
    try:
        load(name, dataset)
        yield name
    finally:
        with connect_admin("postgres") as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def _problem(db: str, dataset: Dataset, key: AnswerKey, case: FixCase) -> str | None:
    """Why the case went wrong, or None. A bad fix refused by the wrong check counts."""
    fault = next(f for f in key.faults if f.type == case.fault_type)
    finding, correction = case.build(dataset, fault)
    result = verify(db, finding, correction)
    if result.check == case.expected_check and result.passed == (case.expected_check == "ok"):
        return None
    return f"expected {case.expected_check}, got {result.check}: {result.reason}"


def main() -> int:
    passed: Counter[str] = Counter()
    failed: Counter[str] = Counter()
    problems: list[str] = []
    for seed in SEEDS:
        dataset, key = build_scenario(scenario_spec(seed))
        with _scratch_database(dataset) as db:
            for case in [*BAD_FIXES, *GOOD_FIXES]:
                problem = _problem(db, dataset, key, case)
                (failed if problem else passed)[case.expected_check] += 1
                if problem:
                    problems.append(f"FAIL seed {seed} {case.name}: {problem}")

    print(f"{'check':<15}{'pass':>6}{'fail':>6}")
    checks = sorted({c.expected_check for c in [*BAD_FIXES, *GOOD_FIXES]})
    for check in checks:
        print(f"{check:<15}{passed[check]:>6}{failed[check]:>6}")
    for problem in problems:
        print(problem)
    print(f"{len(BAD_FIXES)} bad fixes and {len(GOOD_FIXES)} good fixes "
          f"on {len(SEEDS)} seeds: {sum(failed.values())} failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
