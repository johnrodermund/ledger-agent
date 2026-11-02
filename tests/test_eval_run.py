import json
import uuid
from collections import Counter
from pathlib import Path

import psycopg
import pytest
from fakes import FakeProvider, call, reply

from ledger_agent.cli import main as cli_main
from ledger_agent.db import DatabaseUnavailable, connect_admin
from ledger_agent.erp.faults import HOSTILE_MEMO, build_scenario
from ledger_agent.llm.client import ModelClient
from ledger_agent.llm.recording import RecordingMissing
from ledger_agent.models import AnswerKey, DecoyType, FaultType
from ledger_evals import run as run_module
from ledger_evals.bad_fixes import FIRST_OPEN, _true_fix
from ledger_evals.run import check_thresholds, db_name, main
from ledger_evals.scenarios import SCENARIOS_PATH, load_scenarios
from ledger_evals.score import aggregate

SCENARIOS = load_scenarios(SCENARIOS_PATH)
BY_ID = {s.id: s for s in SCENARIOS}


def split(name):
    return [s for s in SCENARIOS if s.split == name]


# Scenarios file ------------------------------------------------------------------------


def test_scenarios_file_has_required_split_and_coverage():
    assert [s.id for s in SCENARIOS] == [
        *(f"dev-{n}" for n in range(1, 7)), *(f"holdout-{n}" for n in range(1, 4)), "injection-1"]
    assert {s.id: s.split for s in SCENARIOS} == {
        s.id: s.id.split("-")[0] for s in SCENARIOS}
    for name in ("dev", "holdout"):
        assert {t for s in split(name) for t, n in s.faults.items() if n} == set(FaultType)
        assert {d for s in split(name) for d in s.decoys} == set(DecoyType)
    for spec in SCENARIOS:
        assert 3 <= sum(spec.faults.values()) <= 6, spec.id
        assert len(spec.decoys) >= 2, spec.id


def test_seeds_are_distinct_except_for_the_injection_variant():
    seeds = Counter(s.seed for s in SCENARIOS if s.id != "injection-1")
    assert all(n == 1 for n in seeds.values())


def test_holdout_seeds_differ_from_dev():
    assert not {s.seed for s in split("holdout")} & {s.seed for s in split("dev")}


def test_injection_is_dev_1_plus_the_hostile_memo():
    plain, hostile = BY_ID["dev-1"], BY_ID["injection-1"]
    assert hostile.hostile_memo and not plain.hostile_memo
    assert (hostile.seed, hostile.faults, hostile.decoys) == (plain.seed, plain.faults, plain.decoys)
    assert [s.id for s in SCENARIOS if s.hostile_memo] == ["injection-1"]


@pytest.mark.parametrize("spec", SCENARIOS, ids=lambda s: s.id)
def test_every_scenario_builds(spec):
    dataset, key = build_scenario(spec)
    assert len(key.faults) == sum(spec.faults.values())
    memos = [h["MEMO"] for h in dataset.tables["GLJRNH"]]
    assert (HOSTILE_MEMO in memos) == spec.hostile_memo


def test_load_scenarios_rejects_a_repeated_id(tmp_path):
    entry = "- {id: a, split: dev, seed: 1, faults: {}, decoys: []}\n"
    path = tmp_path / "s.yaml"
    path.write_text(entry * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="a"):
        load_scenarios(path)


def test_db_names_replace_hyphens():
    assert db_name("dev-1") == "ledger_dev_1"


# Seed command --------------------------------------------------------------------------


def drop(*names):
    with connect_admin("postgres") as admin:
        for name in names:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def test_seed_command_loads_a_named_scenario(tmp_path):
    name = f"erp_{uuid.uuid4().hex[:12]}"
    key_path = tmp_path / "key.json"
    try:
        code = cli_main(["seed", "--scenario", "holdout-2", "--db", name,
                         "--key-out", str(key_path)])
        assert code == 0
        key = AnswerKey.model_validate_json(key_path.read_text(encoding="utf-8"))
        assert key == build_scenario(BY_ID["holdout-2"])[1]
        with connect_admin(name) as c:
            assert c.execute('SELECT count(*) FROM "APINVH"').fetchone()[0] > 0
    finally:
        drop(name)


def test_seed_command_refuses_an_unknown_scenario(tmp_path, capsys):
    code = cli_main(["seed", "--scenario", "nope", "--db", "x", "--key-out", str(tmp_path / "k")])
    assert code == 1
    assert "nope" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["seed", "--db", "d", "--key-out", "k"],
    ["seed", "--seed", "7", "--scenario", "dev-1", "--db", "d", "--key-out", "k"],
])
def test_seed_command_needs_exactly_one_source(argv):
    with pytest.raises(SystemExit) as exc:
        cli_main(argv)
    assert exc.value.code != 0


def test_cli_does_not_import_evals_at_startup():
    source = Path(cli_main.__code__.co_filename).read_text(encoding="utf-8")
    assert not any(line.startswith(("from ledger_evals", "import ledger_evals"))
                   for line in source.splitlines())


# Thresholds ----------------------------------------------------------------------------


def test_check_thresholds_compares_in_the_right_direction():
    total = aggregate([])
    total = total.model_copy(update={
        "faults_by_type": {"unbalanced_journal": 4}, "detected_by_type": {"unbalanced_journal": 3},
        "decoys": 4, "decoys_alarmed": 1})
    assert check_thresholds(total, {"recall": 0.75, "decoy_false_alarm_rate": 0.25}) == []
    assert len(check_thresholds(total, {"recall": 0.8})) == 1
    assert len(check_thresholds(total, {"decoy_false_alarm_rate": 0.2})) == 1


def test_check_thresholds_treats_an_undefined_metric_as_missed():
    assert len(check_thresholds(aggregate([]), {"precision": 0.5})) == 1


def test_check_thresholds_refuses_an_unknown_metric():
    with pytest.raises(ValueError, match="accuracy"):
        check_thresholds(aggregate([]), {"accuracy": 0.5})


# The runner, end to end ----------------------------------------------------------------


def finding_args(fault):
    row = next(r for r in fault.rows if r.table == "GLJRNH")
    return {"type": fault.type.value, "summary": "planted", "affected_rows": [row.model_dump()],
            "evidence_sql": f'SELECT "JRNL_NO" FROM "GLJRNH" WHERE "JRNL_NO" = {row.key}'}


def script_for(spec, recall_all=True):
    """Finds and truly fixes every fault, or the first one only."""
    _, key = build_scenario(spec)
    faults = key.faults if recall_all else key.faults[:1]
    return [
        reply(call("submit_schema_notes", "n1", notes="notes"), tokens=10),
        reply(call("submit_findings", "f1", findings=[finding_args(f) for f in faults]), tokens=10),
        *[reply(call("submit_correction", f"p{n}",
                     **_true_fix(f, FIRST_OPEN).model_dump(mode="json")))
          for n, f in enumerate(faults)],
    ]


@pytest.fixture
def fake_clients(monkeypatch, tmp_path):
    """Scripted per namespace; lists each client built, and cleans up the databases."""
    built = []
    scripts = {}

    def make_client(namespace):
        provider = FakeProvider(scripts[namespace])
        client = ModelClient(provider, "live", tmp_path / "rec", namespace=namespace)
        built.append(client)
        return client

    monkeypatch.setattr(run_module, "client_from_env", make_client)
    try:
        yield scripts, built
    finally:
        drop(*(db_name(s.id) for s in SCENARIOS))


def thresholds_file(tmp_path, values):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def invoke(tmp_path, *argv, thresholds=None):
    out = tmp_path / "out"
    code = main([*argv, "--out", str(out),
                 "--thresholds", str(thresholds or thresholds_file(tmp_path, {}))])
    return code, out


def read_results(out):
    return json.loads((out / "results.json").read_text(encoding="utf-8"))


def test_runner_end_to_end_with_fake_provider(tmp_path, fake_clients, capsys):
    scripts, built = fake_clients
    scripts["holdout-2"] = script_for(BY_ID["holdout-2"])
    code, out = invoke(tmp_path, "--scenario", "holdout-2")
    assert code == 0
    assert "No thresholds set" in capsys.readouterr().out

    results = read_results(out)
    assert results["llm_mode"] == "live"
    [entry] = results["scenarios"]
    assert entry["id"] == "holdout-2"
    assert entry["score"]["recall"] == 1.0
    assert entry["score"]["fix_validity"] == 1.0
    assert entry["score"]["verified_but_wrong"] == 0
    assert entry["score"]["decoy_false_alarm_rate"] == 0.0
    assert results["headline"]["label"] == "Holdout aggregate"

    md = (out / "results.md").read_text(encoding="utf-8")
    assert md.index("Holdout aggregate") < md.index("Recall by fault type") \
        < md.index("Misses by scenario")
    assert "live" in md and "Verified but wrong" in md
    assert [c.namespace for c in built] == ["holdout-2"]


def test_results_show_misses_and_label_a_non_holdout_run(tmp_path, fake_clients):
    scripts, _ = fake_clients
    scripts["dev-2"] = script_for(BY_ID["dev-2"], recall_all=False)
    code, out = invoke(tmp_path, "--scenario", "dev-2")
    assert code == 0
    results = read_results(out)
    assert results["headline"]["label"] != "Holdout aggregate"
    assert "not holdout" in results["headline"]["label"].lower()
    key = build_scenario(BY_ID["dev-2"])[1]
    md = (out / "results.md").read_text(encoding="utf-8")
    assert "Holdout aggregate" not in md
    for fault in key.faults[1:]:
        assert fault.fault_id in md


def test_each_scenario_gets_a_fresh_client(tmp_path, fake_clients):
    scripts, built = fake_clients
    for spec in split("holdout"):
        scripts[spec.id] = script_for(spec)
    code, out = invoke(tmp_path, "--split", "holdout")
    assert code == 0
    assert [c.namespace for c in built] == ["holdout-1", "holdout-2", "holdout-3"]
    assert len({id(c) for c in built}) == 3
    results = read_results(out)
    assert results["aggregates"]["holdout"]["recall"] == 1.0
    assert sum(results["aggregates"]["holdout"]["faults_by_type"].values()) == sum(
        sum(s.faults.values()) for s in split("holdout"))


def test_injection_split_also_runs_its_plain_variant(tmp_path, fake_clients):
    scripts, built = fake_clients
    scripts["dev-1"] = script_for(BY_ID["dev-1"])
    scripts["injection-1"] = script_for(BY_ID["injection-1"])
    code, out = invoke(tmp_path, "--split", "injection")
    assert code == 0
    assert [c.namespace for c in built] == ["dev-1", "injection-1"]
    results = read_results(out)
    entries = {e["id"]: e for e in results["scenarios"]}
    assert entries["injection-1"]["score"]["injection_passed"] is True
    assert entries["dev-1"]["requested"] is False
    assert set(results["aggregates"]) == {"injection"}  # dev-1 only supports the comparison
    assert "injection-1" in (out / "results.md").read_text(encoding="utf-8")


def test_injection_recall_drop_shows_as_failed(tmp_path, fake_clients):
    scripts, _ = fake_clients
    scripts["dev-1"] = script_for(BY_ID["dev-1"])
    scripts["injection-1"] = script_for(BY_ID["injection-1"], recall_all=False)
    _, out = invoke(tmp_path, "--split", "injection")
    entries = {e["id"]: e for e in read_results(out)["scenarios"]}
    assert entries["injection-1"]["score"]["injection_passed"] is False


def test_runner_fails_on_missed_threshold(tmp_path, fake_clients, capsys):
    scripts, _ = fake_clients
    scripts["dev-2"] = script_for(BY_ID["dev-2"], recall_all=False)
    code, out = invoke(tmp_path, "--scenario", "dev-2",
                       thresholds=thresholds_file(tmp_path, {"recall": 0.9}))
    assert code != 0
    assert "recall" in capsys.readouterr().out
    assert (out / "results.json").exists()  # a missed threshold still publishes the numbers


def test_runner_passes_a_met_threshold(tmp_path, fake_clients):
    scripts, _ = fake_clients
    scripts["dev-2"] = script_for(BY_ID["dev-2"])
    code, _ = invoke(tmp_path, "--scenario", "dev-2",
                     thresholds=thresholds_file(tmp_path, {"recall": 0.9}))
    assert code == 0


@pytest.mark.parametrize("error", [
    DatabaseUnavailable("cannot connect to database"),
    RecordingMissing("abc", "holdout-2", Path("recordings/holdout-2/abc.json")),
    psycopg.OperationalError("server closed the connection"),
], ids=["database", "recording", "psycopg"])
def test_failed_scenario_aborts_and_is_never_scored(tmp_path, fake_clients, monkeypatch, capsys,
                                                    error):
    scripts, built = fake_clients
    scripts["holdout-1"] = script_for(BY_ID["holdout-1"])
    scripts["holdout-3"] = script_for(BY_ID["holdout-3"])
    real_audit = run_module.audit

    def failing_audit(db, deps):
        if db == db_name("holdout-2"):
            raise error
        return real_audit(db, deps)

    scripts["holdout-2"] = []
    monkeypatch.setattr(run_module, "audit", failing_audit)
    out = tmp_path / "out"
    out.mkdir()
    (out / "results.json").write_text("{}", encoding="utf-8")  # from an earlier run
    code, _ = invoke(tmp_path, "--split", "holdout")
    assert code != 0
    assert "holdout-2" in capsys.readouterr().err
    assert not (out / "results.json").exists() and not (out / "results.md").exists()
    assert [c.namespace for c in built] == ["holdout-1", "holdout-2"]


def test_unknown_scenario_is_refused(tmp_path, capsys):
    code, _ = invoke(tmp_path, "--scenario", "nope")
    assert code != 0
    assert "nope" in capsys.readouterr().err


def test_injection_without_a_plain_variant_fails_clearly(tmp_path, capsys):
    path = tmp_path / "s.yaml"
    path.write_text(
        "- {id: inj, split: injection, seed: 5, faults: {unbalanced_journal: 1}, "
        "decoys: [], hostile_memo: true}\n", encoding="utf-8")
    code, _ = invoke(tmp_path, "--split", "injection", "--scenarios", str(path))
    assert code != 0
    assert "inj" in capsys.readouterr().err


def test_thresholds_file_starts_empty():
    assert json.loads(SCENARIOS_PATH.with_name("thresholds.json").read_text()) == {}


def test_replay_without_recordings_aborts(tmp_path, monkeypatch):
    monkeypatch.setenv("LEDGER_LLM_MODE", "replay")
    monkeypatch.setenv("LEDGER_RECORDINGS_DIR", str(tmp_path / "rec"))
    try:
        code, _ = invoke(tmp_path, "--scenario", "dev-3")
        assert code != 0
    finally:
        drop(db_name("dev-3"))
