"""The eval scenarios, defined in evals/scenarios.yaml."""

from collections import Counter
from pathlib import Path

import yaml

from ledger_agent.models import ScenarioSpec

SCENARIOS_PATH = Path(__file__).resolve().parents[2] / "evals" / "scenarios.yaml"


def load_scenarios(path: Path = SCENARIOS_PATH) -> list[ScenarioSpec]:
    specs = [ScenarioSpec.model_validate(raw)
             for raw in yaml.safe_load(Path(path).read_text(encoding="utf-8"))]
    repeated = [name for name, n in Counter(s.id for s in specs).items() if n > 1]
    if repeated:
        raise ValueError(f"repeated scenario id: {', '.join(repeated)}")
    return specs
