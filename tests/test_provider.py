"""AnthropicProvider against the real SDK over a stubbed HTTP transport. Nothing leaves
the process, and the key is a placeholder that is never checked."""

import json

import anthropic
import httpx
import pytest

from ledger_agent.llm.client import (
    AnthropicProvider,
    ModelRequest,
    ModelResponse,
    ToolCall,
    TransientProviderError,
    assistant_message,
    tool_result_message,
)

TOOL = {"name": "run_sql", "description": "q", "input_schema": {"type": "object"}}
REQ = ModelRequest("claude-haiku-5-5", "sys", [{"role": "user", "content": "hi"}], [TOOL])


def message(content, stop_reason="end_turn"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": REQ.model,
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 7}}


def provider_with(handler):
    provider = AnthropicProvider(api_key="placeholder")
    provider._client = anthropic.Anthropic(
        api_key="placeholder", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return provider


def replying(body, status=200, headers=None):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(status, json=body, headers=headers)

    return provider_with(handler), sent


def test_haiku_sends_disabled_thinking_and_no_tool_choice():
    provider, sent = replying(message([{"type": "text", "text": "ok"}]))
    provider.complete(REQ)
    assert sent[0]["thinking"] == {"type": "disabled"}
    assert "tool_choice" not in sent[0] and "output_config" not in sent[0]
    assert sent[0]["model"] == "claude-haiku-5-5" and sent[0]["tools"] == [TOOL]


def test_sonnet_sends_between_tools_thinking():
    provider, sent = replying(message([{"type": "text", "text": "ok"}]))
    provider.complete(ModelRequest("claude-sonnet-5-5", "sys", REQ.messages, [TOOL]))
    assert sent[0]["thinking"] == {"type": "between_tools"}


def test_a_model_without_a_thinking_setting_is_refused_before_any_call():
    provider, sent = replying(message([]))
    with pytest.raises(ValueError, match="no thinking setting"):
        provider.complete(ModelRequest("claude-unknown", "sys", REQ.messages, [TOOL]))
    assert sent == []


def test_a_forced_turn_steers_from_the_system_prompt():
    provider, sent = replying(message([]))
    provider.complete(ModelRequest(REQ.model, "sys", REQ.messages, [TOOL], "run_sql"))
    assert "tool_choice" not in sent[0] and "`run_sql`" in sent[0]["system"][0]["text"]


def test_parses_text_tool_calls_usage_and_stop_reason():
    provider, _ = replying(message([
        {"type": "text", "text": "a"},
        {"type": "tool_use", "id": "t1", "name": "run_sql", "input": {"sql": "select 1"}},
        {"type": "text", "text": "b"}], "tool_use"))
    assert provider.complete(REQ) == ModelResponse(
        text="ab", tool_calls=[ToolCall("t1", "run_sql", {"sql": "select 1"})],
        input_tokens=11, output_tokens=7, stop_reason="tool_use")


def test_keeps_thinking_blocks_and_passes_them_back_unchanged():
    blocks = [
        {"type": "thinking", "thinking": "checking balances", "signature": "s1"},
        {"type": "tool_use", "id": "t1", "name": "run_sql", "input": {"sql": "select 1"}},
        {"type": "thinking", "thinking": "now duplicates", "signature": "s2"},
        {"type": "tool_use", "id": "t2", "name": "run_sql", "input": {"sql": "select 2"}}]
    provider, _ = replying(message(blocks, "tool_use"))
    resp = provider.complete(REQ)
    assert resp.text == "" and [c.id for c in resp.tool_calls] == ["t1", "t2"]
    assert assistant_message(resp) == {"role": "assistant", "content": blocks}


def test_a_reply_without_thinking_keeps_the_built_message():
    provider, _ = replying(message([
        {"type": "tool_use", "id": "t1", "name": "run_sql", "input": {"sql": "select 1"}},
        {"type": "text", "text": "done"}], "tool_use"))
    resp = provider.complete(REQ)
    assert resp.content is None
    assert [b["type"] for b in assistant_message(resp)["content"]] == ["text", "tool_use"]


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal", "end_turn"])
def test_reports_the_stop_reason(stop_reason):
    provider, _ = replying(message([], stop_reason))
    assert provider.complete(REQ).stop_reason == stop_reason


def test_tool_results_round_trip_in_the_api_shape():
    resp = ModelResponse(text="", tool_calls=[ToolCall("t1", "run_sql", {"sql": "select 1"})])
    history = [*REQ.messages, assistant_message(resp),
               tool_result_message([("t1", "rows", False), ("t2", "bad", True)])]
    provider, sent = replying(message([{"type": "text", "text": "ok"}]))
    provider.complete(ModelRequest(REQ.model, "sys", history, [TOOL]))
    assert sent[0]["messages"][1]["content"] == [
        {"type": "tool_use", "id": "t1", "name": "run_sql", "input": {"sql": "select 1"}}]
    assert sent[0]["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "rows", "is_error": False},
        {"type": "tool_result", "tool_use_id": "t2", "content": "bad", "is_error": True}]


ERROR = {"type": "error", "error": {"type": "api_error", "message": "boom"}}


@pytest.mark.parametrize("status", [429, 500, 529])
def test_rate_limits_and_server_errors_are_transient(status):
    provider, _ = replying(ERROR, status)
    with pytest.raises(TransientProviderError):
        provider.complete(REQ)


def test_connection_errors_are_transient():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(TransientProviderError):
        provider_with(handler).complete(REQ)


@pytest.mark.parametrize("status", [400, 401, 404])
def test_client_errors_are_not_transient(status):
    provider, _ = replying(ERROR, status)
    with pytest.raises(anthropic.APIStatusError):
        provider.complete(REQ)


@pytest.mark.parametrize("headers, expected", [
    ({"retry-after": "42"}, 42.0),
    ({"retry-after": "1.5"}, 1.5),
    ({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, None),
    ({}, None),
])
def test_retry_after_is_carried_on_the_error(headers, expected):
    provider, _ = replying(ERROR, 429, headers)
    with pytest.raises(TransientProviderError) as exc:
        provider.complete(REQ)
    assert exc.value.retry_after == expected


def test_marks_the_system_prompt_and_the_conversation_for_caching():
    provider, sent = replying(message([{"type": "text", "text": "ok"}]))
    provider.complete(REQ)
    assert sent[0]["system"] == [
        {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    assert sent[0]["cache_control"] == {"type": "ephemeral"}


def test_reads_cache_usage_and_treats_a_missing_count_as_zero():
    body = message([{"type": "text", "text": "ok"}])
    body["usage"] |= {"cache_creation_input_tokens": 300, "cache_read_input_tokens": 900}
    provider, _ = replying(body)
    resp = provider.complete(REQ)
    assert (resp.input_tokens, resp.cache_write_tokens, resp.cache_read_tokens) == (11, 300, 900)
    provider, _ = replying(message([{"type": "text", "text": "ok"}]))
    resp = provider.complete(REQ)
    assert (resp.cache_write_tokens, resp.cache_read_tokens) == (0, 0)
