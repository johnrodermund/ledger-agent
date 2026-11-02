"""Command line entry point."""

import argparse
import sys
from pathlib import Path

import psycopg

from ledger_agent.agent.graph import audit
from ledger_agent.agent.state import Deps
from ledger_agent.db import DatabaseUnavailable, connect_ro
from ledger_agent.erp.faults import build_scenario
from ledger_agent.erp.load import load
from ledger_agent.llm.client import TransientProviderError, client_from_env
from ledger_agent.llm.recording import RecordingMissing
from ledger_agent.models import DecoyType, FaultType, ScenarioSpec
from ledger_agent.report import write_report


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


def default_namespace(dbname: str) -> str:
    """ledger_dev_1 gives dev-1: the recordings namespace follows the database name."""
    return dbname.removeprefix("ledger_").replace("_", "-")


def _failure_types() -> tuple[type[BaseException], ...]:
    types: tuple[type[BaseException], ...] = (
        DatabaseUnavailable, psycopg.Error, RecordingMissing, TransientProviderError, ValueError, RuntimeError,
        KeyError)
    try:
        import anthropic
    except ImportError:
        return types
    return (*types, anthropic.APIError)


def _audit(args: argparse.Namespace) -> int:
    try:
        connect_ro(args.db).close()  # run_query reports connection errors to the model
        client = client_from_env(args.namespace or default_namespace(args.db))
        result = audit(args.db, Deps(client))
        write_report(result, args.out)
    except _failure_types() as e:
        message = " ".join(str(e).split()) or type(e).__name__
        if isinstance(e, KeyError):
            message = f"missing environment variable {message}"
        print(f"error: {message}", file=sys.stderr)
        return 1
    print(f"Audit complete: report written to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ledger-agent")
    commands = parser.add_subparsers(dest="command", required=True)

    seed = commands.add_parser("seed", help="load a planted scenario into a database")
    seed.add_argument("--seed", type=int, required=True, help="scenario seed")
    seed.add_argument("--db", required=True, help="database to drop and recreate")
    seed.add_argument("--key-out", type=Path, required=True, help="answer key output path")
    seed.set_defaults(run=_seed)

    audit_cmd = commands.add_parser("audit", help="audit a database and write a report")
    audit_cmd.add_argument("--db", required=True, help="database to audit")
    audit_cmd.add_argument("--out", type=Path, required=True, help="report output directory")
    audit_cmd.add_argument("--namespace", help="recordings namespace (default: from --db)")
    audit_cmd.set_defaults(run=_audit)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
