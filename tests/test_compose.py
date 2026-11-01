from pathlib import Path

import pytest
import yaml

from ledger_agent.db import connect_admin


def _service(name):
    # safe_load resolves anchors and merge keys, like compose does.
    return yaml.safe_load(Path("docker-compose.yml").read_text())["services"][name]


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
