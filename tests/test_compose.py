from pathlib import Path

import pytest
import yaml

from ledger_agent.db import connect_admin


def _service(name):
    # safe_load resolves anchors and merge keys, like compose does.
    return yaml.safe_load(Path("docker-compose.yml").read_text())["services"][name]


def test_postgres_runs_without_autovacuum():
    assert "autovacuum=off" in _service("db")["command"]


def test_agent_service_gets_no_admin_password():
    agent = _service("agent")
    assert "env_file" not in agent
    assert "POSTGRES_PASSWORD" not in agent["environment"]
    assert {"LEDGER_RO_PASSWORD", "LEDGER_RW_PASSWORD"} <= set(agent["environment"])


def test_connect_admin_fails_clearly_without_password(monkeypatch):
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="POSTGRES_PASSWORD"):
        connect_admin("postgres")


def test_healthcheck_uses_tcp():
    assert "-h 127.0.0.1" in _service("db")["healthcheck"]["test"][1]


def test_agent_gets_the_llm_settings_and_no_keys():
    agent = _service("agent")
    assert {"LEDGER_DB_HOST", "LEDGER_DB_PORT", "LEDGER_LLM_MODE", "ANTHROPIC_API_KEY"} \
        <= set(agent["environment"])
    assert agent["environment"]["LEDGER_LLM_MODE"].startswith("${LEDGER_LLM_MODE:-replay}")
    assert not any("keys" in v for v in agent["volumes"])
    assert any(v.startswith("./out:") for v in agent["volumes"])
    assert any(v.startswith("./recordings:") for v in agent["volumes"])


def test_seed_loads_dev_1_and_writes_its_key():
    seed = _service("seed")
    assert "ledger_agent.cli seed --scenario dev-1 --db ledger_dev_1" in seed["command"][-1]
    assert "--key-out /keys/dev-1.json" in seed["command"][-1]
    assert any(v.startswith("./evals:") and v.endswith(":ro") for v in seed["volumes"])


def test_services_run_in_order():
    seed, agent, evals = (_service(n) for n in ("seed", "agent", "evals"))
    assert seed["depends_on"]["db"]["condition"] == "service_healthy"
    assert agent["depends_on"]["seed"]["condition"] == "service_healthy"
    assert agent["depends_on"]["db"]["condition"] == "service_healthy"
    assert evals["depends_on"]["agent"]["condition"] == "service_healthy"


def test_evals_runs_guardrails_then_all_splits_without_keys_volume():
    evals = _service("evals")
    command = " ".join(evals["command"])
    assert command.index("ledger_evals.guardrails") < command.index("ledger_evals.run")
    assert "--split all --out /out/evals" in command
    assert not any("keys" in v for v in evals["volumes"])
    assert any(v.startswith("./out:") for v in evals["volumes"])
    assert any(v.startswith("./recordings:") for v in evals["volumes"])


def test_only_admin_services_get_the_admin_password():
    for name in ("seed", "evals"):
        assert "POSTGRES_PASSWORD" in _service(name)["environment"]
    assert "POSTGRES_PASSWORD" not in _service("agent")["environment"]


def test_one_shot_services_idle_and_report_health_from_a_fresh_marker():
    for name in ("seed", "agent"):
        service = _service(name)
        command = service["command"][-1]
        assert command.startswith("rm -f /tmp/done && ")
        assert command.endswith("&& touch /tmp/done && exec sleep infinity")
        assert "/tmp" in service["tmpfs"]  # a restart cannot keep the old marker
        health = service["healthcheck"]
        assert health["test"][1] == "test -f /tmp/done"
        interval = int(health["interval"].rstrip("s"))
        assert interval * health["retries"] >= 45 * 60  # live workflow timeout-minutes
