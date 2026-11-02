"""Rules about the repository itself: README claims, workflows and committed values."""

import re
from pathlib import Path

import yaml

README = Path("README.md")
WORKFLOWS = Path(".github/workflows")


def _compose():
    return yaml.safe_load(Path("docker-compose.yml").read_text())


def _workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _readme():
    return README.read_text(encoding="utf-8")


def test_agent_service_has_no_keys_volume():
    assert "keys" not in str(_compose()["services"]["agent"].get("volumes", []))


def test_no_secret_is_committed_in_env_example():
    for line in Path(".env.example").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, _, value = line.partition("=")
        assert name != "ANTHROPIC_API_KEY" or value == ""
        assert not value.startswith("sk-"), name
        if name.endswith("PASSWORD"):
            assert value.endswith("_test_pw"), name


def test_readme_states_default_runs_are_replays():
    text = _readme().lower()
    assert "replay" in text
    assert "default" in text
    assert "no api key" in text


def test_readme_results_section_comes_first():
    headings = re.findall(r"^#{2,} (.+)$", _readme(), flags=re.M)
    assert headings[0].lower().startswith("results")
    first_title = _readme().splitlines()[0]
    assert first_title.startswith("# ")


def test_readme_results_make_no_numeric_claims_until_recorded():
    section = _readme().split("## Results", 1)[1].split("\n## ", 1)[0]
    assert "pending" in section.lower()


def test_readme_limitations_are_disclosed():
    text = _readme().lower()
    for phrase in ("overlap", "five", "verified_but_wrong", "synthetic", "never applied",
                   "does not prove", "token limit", "timeout-minutes"):
        assert phrase in text, phrase


def test_readme_discloses_that_the_type_names_are_given_to_the_model():
    limitations = _readme().split("## Limitations", 1)[1].split("\n## ", 1)[0].lower()
    assert "five" in limitations and "type names" in limitations


def test_readme_hundred_percent_applies_only_to_guardrails():
    for line in _readme().splitlines():
        if "100%" in line:
            assert "guardrail" in line.lower(), line


def test_readme_has_graph_and_extensions():
    text = _readme()
    assert "```mermaid" in text
    assert "apply" in text and "DSPy" in text


def test_ci_runs_tests_guardrails_and_replay_evals_on_push():
    wf = _workflow("ci.yml")
    assert wf.get(True, wf.get("on")) is not None and "push" in (wf.get(True) or wf["on"])
    assert wf["permissions"] == {"contents": "read"}
    steps = " ".join(s.get("run", "") for j in wf["jobs"].values() for s in j["steps"])
    assert "cp .env.example .env" in steps
    assert "pytest" in steps
    assert "ledger_evals.guardrails" in steps
    # The evals service runs ledger_evals.run, so the replay evals are the compose run.
    assert "up --build --abort-on-container-exit --exit-code-from evals" in steps


def test_workflow_actions_are_pinned_to_major_versions():
    for path in WORKFLOWS.glob("*.yml"):
        for use in re.findall(r"uses:\s*(\S+)", path.read_text()):
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@v\d+", use), (path, use)


def test_live_evals_workflow():
    wf = _workflow("live-evals.yml")
    triggers = wf.get(True) or wf["on"]
    assert "schedule" in triggers
    mode = triggers["workflow_dispatch"]["inputs"]["mode"]
    assert mode["type"] == "choice" and mode["options"] == ["live", "record"]
    assert mode["default"] == "live"
    assert wf["permissions"] == {"contents": "read"}
    text = (WORKFLOWS / "live-evals.yml").read_text()
    assert "secrets.ANTHROPIC_API_KEY" in text
    assert "timeout-minutes" in text
    assert "upload-artifact" in text and "recordings" in text
    assert "git commit" not in text and "git push" not in text


def test_live_evals_scopes_the_key_to_the_eval_step_and_checks_it():
    wf = _workflow("live-evals.yml")
    job = wf["jobs"]["evals"]
    assert "ANTHROPIC_API_KEY" not in job.get("env", {})
    holders = [s for s in job["steps"] if "ANTHROPIC_API_KEY" in s.get("env", {})]
    assert len(holders) == 1
    run = holders[0]["run"]
    assert run.index('-z "$ANTHROPIC_API_KEY"') < run.index("docker compose up")


def test_readme_says_the_default_run_needs_recordings():
    assert "recordings" in _readme().split("Services start in this order")[0].lower()
