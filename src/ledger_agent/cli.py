"""Command line entry point."""

import argparse
import sys
from pathlib import Path

from ledger_agent.erp.faults import build_scenario
from ledger_agent.erp.load import load
from ledger_agent.models import DecoyType, FaultType, ScenarioSpec


def _seed(args: argparse.Namespace) -> int:
    """Load one of each fault and decoy into --db. The key goes to --key-out only."""
    spec = ScenarioSpec(id=f"seed-{args.seed}", split="dev", seed=args.seed,
                        faults={t: 1 for t in FaultType}, decoys=list(DecoyType))
    dataset, key = build_scenario(spec)
    load(args.db, dataset)
    args.key_out.parent.mkdir(parents=True, exist_ok=True)
    args.key_out.write_text(key.to_json(), encoding="utf-8")
    print(f"Seeded database {args.db}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ledger-agent")
    commands = parser.add_subparsers(dest="command", required=True)

    seed = commands.add_parser("seed", help="load a planted scenario into a database")
    seed.add_argument("--seed", type=int, required=True, help="scenario seed")
    seed.add_argument("--db", required=True, help="database to drop and recreate")
    seed.add_argument("--key-out", type=Path, required=True, help="answer key output path")
    seed.set_defaults(run=_seed)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
