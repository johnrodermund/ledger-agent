import json

import pytest

from ledger_agent.llm.client import (
    API_MAX_RETRIES,
    BACKOFF_BASE_SECONDS,
    MAX_RETRY_WAIT_SECONDS,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ToolCall,
    TransientProviderError,
    client_from_env,
)
from ledger_agent.llm.recording import RecordingMissing, request_key

TOOLS = [{"name": "run_sql", "description": "q", "input_schema": {"type": "object"}}]
REQ = ModelRequest(
    model="m1",
    system="sys",
    messages=[{"role": "user", "content": "hi"}],
    tools=TOOLS,
    tool_choice=None,
)
RESP = ModelResponse(
    text="recorded",
    tool_calls=[ToolCall(id="t1", name="run_sql", arguments={"sql": "select 1"})],
    input_tokens=10,
    output_tokens=5,
    stop_reason="tool_use",
)


class FakeProvider:
    def __init__(self, response=RESP):
        self.response = response
        self.calls = 0

    def complete(self, req):
        self.calls += 1
        return self.response


class ExplodingProvider:
    def complete(self, req):
        raise AssertionError("provider must not be called")


def seeded(tmp_path, namespace="ns"):
    ModelClient(FakeProvider(), "record", tmp_path, namespace=namespace).complete(REQ)
    return tmp_path


def test_record_then_replay_returns_same_response(tmp_path):
    provider = FakeProvider()
    recorded = ModelClient(provider, "record", tmp_path, namespace="ns").complete(REQ)
    replayed = ModelClient(ExplodingProvider(), "replay", tmp_path, namespace="ns").complete(REQ)
    assert provider.calls == 1
    assert replayed == recorded == RESP


def test_replay_never_calls_provider(tmp_path):
    client = ModelClient(ExplodingProvider(), "replay", seeded(tmp_path), namespace="ns")
    assert client.complete(REQ).text == "recorded"


def test_replay_miss_raises(tmp_path):
    with pytest.raises(RecordingMissing) as exc:
        ModelClient(ExplodingProvider(), "replay", tmp_path, namespace="ns").complete(REQ)
    msg = str(exc.value)
    assert request_key(REQ) in msg
    assert "ns" in msg
    assert "LEDGER_LLM_MODE=record" in msg


def test_key_changes_with_model_messages_or_tools():
    base = request_key(REQ)
    assert request_key(ModelRequest("m2", "sys", REQ.messages, TOOLS, None)) != base
    other_msgs = [{"role": "user", "content": "bye"}]
    assert request_key(ModelRequest("m1", "sys", other_msgs, TOOLS, None)) != base
    assert request_key(ModelRequest("m1", "sys", REQ.messages, [], None)) != base
    assert request_key(ModelRequest("m1", "other", REQ.messages, TOOLS, None)) != base
    assert request_key(ModelRequest("m1", "sys", REQ.messages, TOOLS, "run_sql")) != base


def test_key_ignores_dict_ordering():
    a = ModelRequest("m", "s", [{"role": "user", "content": "x"}], [{"name": "a", "description": "d", "input_schema": {"b": 1, "a": 2}}], None)
    b = ModelRequest("m", "s", [{"content": "x", "role": "user"}], [{"input_schema": {"a": 2, "b": 1}, "description": "d", "name": "a"}], None)
    assert request_key(a) == request_key(b)


def test_tokens_accumulate_in_replay_too(tmp_path):
    seeded(tmp_path)
    client = ModelClient(ExplodingProvider(), "replay", tmp_path, namespace="ns")
    client.complete(REQ)
    client.complete(REQ)
    assert client.input_tokens == 20
    assert client.output_tokens == 10
    assert client.tokens_used == 30
    assert client.calls == [(10, 5, 0, 0), (10, 5, 0, 0)]


def test_retries_transient_errors_then_raises(tmp_path):
    class Flaky:
        def __init__(self, failures):
            self.failures, self.calls = failures, 0

        def complete(self, req):
            self.calls += 1
            if self.calls <= self.failures:
                raise TransientProviderError("429")
            return RESP

    sleeps = []
    ok = Flaky(API_MAX_RETRIES)
    client = ModelClient(ok, "live", tmp_path, sleep=sleeps.append)
    assert client.complete(REQ) == RESP
    assert ok.calls == API_MAX_RETRIES + 1
    assert sleeps == sorted(sleeps) and len(set(sleeps)) == len(sleeps) == API_MAX_RETRIES

    bad = Flaky(API_MAX_RETRIES + 1)
    with pytest.raises(TransientProviderError):
        ModelClient(bad, "live", tmp_path, sleep=lambda s: None).complete(REQ)
    assert bad.calls == API_MAX_RETRIES + 1


def test_backoff_allows_at_least_six_attempts_over_several_minutes(tmp_path):
    class Down:
        calls = 0

        def complete(self, req):
            self.calls += 1
            raise TransientProviderError("overloaded")

    sleeps, down = [], Down()
    with pytest.raises(TransientProviderError):
        ModelClient(down, "live", tmp_path, sleep=sleeps.append).complete(REQ)
    assert down.calls >= 6
    assert 180 <= sum(sleeps) <= 600


class RetryAfter:
    def __init__(self, seconds):
        self.seconds, self.calls = seconds, 0

    def complete(self, req):
        self.calls += 1
        if self.calls == 1:
            raise TransientProviderError("429", retry_after=self.seconds)
        return RESP


def test_retry_after_replaces_a_shorter_backoff(tmp_path):
    sleeps = []
    ModelClient(RetryAfter(90.0), "live", tmp_path, sleep=sleeps.append).complete(REQ)
    assert sleeps == [90.0]


def test_a_retry_after_shorter_than_the_backoff_does_not_shorten_it(tmp_path):
    sleeps = []
    ModelClient(RetryAfter(0.1), "live", tmp_path, sleep=sleeps.append).complete(REQ)
    assert sleeps == [BACKOFF_BASE_SECONDS]


def test_an_absurd_retry_after_is_capped(tmp_path):
    sleeps = []
    ModelClient(RetryAfter(86400.0), "live", tmp_path, sleep=sleeps.append).complete(REQ)
    assert sleeps == [MAX_RETRY_WAIT_SECONDS]


def test_non_transient_errors_are_not_retried(tmp_path):
    class Broken:
        calls = 0

        def complete(self, req):
            self.calls += 1
            raise ValueError("bad request")

    p = Broken()
    with pytest.raises(ValueError):
        ModelClient(p, "live", tmp_path, sleep=lambda s: None).complete(REQ)
    assert p.calls == 1


def test_live_mode_writes_nothing(tmp_path):
    ModelClient(FakeProvider(), "live", tmp_path, namespace="ns").complete(REQ)
    assert list(tmp_path.rglob("*")) == []


def test_recording_file_is_deterministic_and_secret_free(tmp_path):
    seeded(tmp_path)
    path = tmp_path / "ns" / f"{request_key(REQ)}.json"
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    data = json.loads(text)
    assert text == json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert data["response"]["tool_calls"][0]["arguments"] == {"sql": "select 1"}
    assert data["response"]["stop_reason"] == "tool_use"
    assert "api_key" not in text.lower() and "x-api-key" not in text.lower()


def test_recording_holds_the_key_response_model_and_metadata_only(tmp_path):
    seeded(tmp_path)
    text = (tmp_path / "ns" / f"{request_key(REQ)}.json").read_text(encoding="utf-8")
    data = json.loads(text)
    assert set(data) == {"key", "model", "response", "metadata"}
    assert data["key"] == request_key(REQ) and data["model"] == "m1"
    assert data["metadata"] == {"namespace": "ns", "message_count": 1, "tools": ["run_sql"],
                                "tool_choice": None}
    assert '"sys"' not in text and '"hi"' not in text


def test_record_reuses_an_existing_recording_without_calling_the_api(tmp_path):
    seeded(tmp_path)
    path = tmp_path / "ns" / f"{request_key(REQ)}.json"
    before = path.read_text(encoding="utf-8")
    client = ModelClient(ExplodingProvider(), "record", tmp_path, namespace="ns")
    assert client.complete(REQ) == RESP
    assert client.tokens_used == 15  # a resumed run still counts what it replays
    assert path.read_text(encoding="utf-8") == before


def test_record_calls_the_api_for_a_request_it_has_not_seen(tmp_path):
    seeded(tmp_path)
    provider = FakeProvider()
    other = ModelRequest("m1", "sys", [{"role": "user", "content": "bye"}], TOOLS, None)
    ModelClient(provider, "record", tmp_path, namespace="ns").complete(other)
    assert provider.calls == 1


def test_a_failed_write_leaves_no_partial_recording(tmp_path, monkeypatch):
    from pathlib import Path

    def failing_replace(self, target):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(OSError):
        seeded(tmp_path)
    assert not (tmp_path / "ns" / f"{request_key(REQ)}.json").exists()


def test_client_from_env_defaults_to_replay_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("LEDGER_LLM_MODE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LEDGER_RECORDINGS_DIR", str(tmp_path))
    seeded(tmp_path, namespace="demo")
    assert client_from_env("demo").complete(REQ).text == "recorded"


def test_client_from_env_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv("LEDGER_LLM_MODE", "bogus")
    with pytest.raises(ValueError):
        client_from_env("demo")


class CachingProvider:
    def complete(self, req):
        return ModelResponse(text="ok", input_tokens=10, output_tokens=5,
                             cache_write_tokens=100, cache_read_tokens=1_000)


def test_budgets_count_the_whole_prompt_and_calls_keep_the_cache_split(tmp_path):
    client = ModelClient(CachingProvider(), "record", tmp_path, namespace="ns")
    client.complete(REQ)
    assert client.input_tokens == 1_110 and client.tokens_used == 1_115
    assert client.calls == [(10, 5, 100, 1_000)]
    replay = ModelClient(ExplodingProvider(), "replay", tmp_path, namespace="ns")
    replay.complete(REQ)
    assert replay.calls == [(10, 5, 100, 1_000)]


def test_a_recording_made_before_caching_replays_with_no_cache_counts(tmp_path):
    seeded(tmp_path)
    path = next(tmp_path.rglob("*.json"))
    doc = json.loads(path.read_text(encoding="utf-8"))
    for name in ("cache_write_tokens", "cache_read_tokens"):
        doc["response"].pop(name, None)
    path.write_text(json.dumps(doc), encoding="utf-8")
    client = ModelClient(ExplodingProvider(), "replay", tmp_path, namespace="ns")
    client.complete(REQ)
    assert client.calls == [(10, 5, 0, 0)]
