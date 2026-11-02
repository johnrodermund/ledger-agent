import uuid

import pytest

from ledger_agent.db import connect_admin
from ledger_agent.erp.faults import build_scenario
from ledger_agent.erp.generate import N_CLOSED, N_PERIODS
from ledger_agent.erp.load import load
from ledger_agent.models import FaultType
from ledger_agent.verifier.verify import VerifyResult, verify
from ledger_evals import guardrails
from ledger_evals.bad_fixes import BAD_FIXES, GOOD_FIXES

SEEDS = guardrails.SEEDS


@pytest.fixture(scope="module", params=SEEDS)
def scenario(request):
    name = f"guard_{uuid.uuid4().hex[:12]}"
    dataset, key = build_scenario(guardrails.scenario_spec(request.param))
    load(name, dataset)
    yield name, dataset, key
    with connect_admin("postgres") as admin:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def fault_for(key, case):
    return next(f for f in key.faults if f.type == case.fault_type)


@pytest.mark.parametrize("case", BAD_FIXES, ids=lambda c: c.name)
def test_every_bad_fix_is_rejected_at_expected_check(case, scenario):
    db, dataset, key = scenario
    finding, fix = case.build(dataset, fault_for(key, case))
    result = verify(db, finding, fix)
    assert (result.passed, result.check) == (False, case.expected_check), result


@pytest.mark.parametrize("case", GOOD_FIXES, ids=lambda c: c.name)
def test_every_good_fix_is_accepted(case, scenario):
    db, dataset, key = scenario
    finding, fix = case.build(dataset, fault_for(key, case))
    result = verify(db, finding, fix)
    assert (result.passed, result.check) == (True, case.expected_check), result


def test_bad_fix_library_covers_every_check_and_fault():
    assert {c.expected_check for c in BAD_FIXES} >= {
        "structure", "change_limits", "resolution", "collateral"}
    assert {c.fault_type for c in BAD_FIXES} == set(FaultType)
    assert len(BAD_FIXES) >= 15


def test_good_fix_library_covers_every_fault():
    assert {c.fault_type for c in GOOD_FIXES} == set(FaultType)
    assert {c.expected_check for c in GOOD_FIXES} == {"ok"}


def test_case_names_are_unique():
    names = [c.name for c in [*BAD_FIXES, *GOOD_FIXES]]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("seed", SEEDS)
def test_good_fixes_only_post_to_the_open_periods(seed):
    dataset, key = build_scenario(guardrails.scenario_spec(seed))
    closed = {p["PRD_ID"] for p in dataset.tables["GLPERD"] if p["STAT_CD"] == "C"}
    open_periods = {p["PRD_ID"] for p in dataset.tables["GLPERD"] if p["STAT_CD"] == "O"}
    assert open_periods == set(range(N_CLOSED + 1, N_PERIODS + 1))
    assert len(open_periods) == 2

    used = set()
    for case in GOOD_FIXES:
        _, fix = case.build(dataset, fault_for(key, case))
        periods = {ln.prd_id for ln in fix.lines}
        assert periods and periods.isdisjoint(closed)
        used |= periods
    assert used == open_periods


def test_cli_passes_on_the_real_verifier(capsys):
    assert guardrails.main() == 0
    printed = capsys.readouterr().out
    for check in ("structure", "change_limits", "resolution", "collateral", "ok"):
        assert check in printed
    assert "FAIL" not in printed


def test_cli_fails_when_the_verifier_accepts_everything(monkeypatch, capsys):
    monkeypatch.setattr(guardrails, "verify",
                        lambda db, finding, fix: VerifyResult(True, "ok", "accepted"))
    assert guardrails.main() == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_fails_when_the_verifier_rejects_everything(monkeypatch, capsys):
    monkeypatch.setattr(guardrails, "verify",
                        lambda db, finding, fix: VerifyResult(False, "structure", "refused"))
    assert guardrails.main() == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_drops_its_scratch_database():
    guardrails.main()
    with connect_admin("postgres") as admin:
        left = admin.execute("SELECT count(*) FROM pg_database WHERE datname LIKE %s",
                             (f"{guardrails.DB_PREFIX}%",)).fetchone()[0]
    assert left == 0
