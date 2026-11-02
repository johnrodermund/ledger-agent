"""Model client with record and replay.

Messages are plain JSON-serialisable dicts in the Anthropic content-block shape
and tools are ``{name, description, input_schema}``. Providers convert to and
from their SDK, so no SDK object ever enters a message and request hashes stay
stable across record and replay.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from ledger_agent.llm.recording import load_recording, recording_exists, save_recording

DEFAULT_MODEL = "claude-haiku-5-5"
API_MAX_RETRIES = 6  # seven attempts, about five minutes of backoff in all
BACKOFF_BASE_SECONDS = 5.0
MAX_RETRY_WAIT_SECONDS = 600.0
DEFAULT_MAX_TOKENS = 16000
# Thinking is off on every model, so no thinking block is ever passed back. Each model
# spells that differently and refuses the other's spelling.
THINKING_BY_MODEL = {
    "claude-haiku-5-5": {"type": "disabled"},  # accepted at the default effort, medium
    "claude-sonnet-5-5": {"type": "between_tools"},
}

Mode = Literal["replay", "record", "live"]
MODES = ("replay", "record", "live")


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelRequest:
    model: str
    system: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    tool_choice: str | None = None  # None: model's choice; a tool name: force that tool


@dataclass
class ModelResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None


class TransientProviderError(Exception):
    """Rate limit, server or connection failure that is worth retrying. retry_after is
    the wait in seconds the server asked for, if it asked."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class Provider(Protocol):
    def complete(self, req: ModelRequest) -> ModelResponse: ...


def assistant_message(resp: ModelResponse) -> dict[str, Any]:
    """The neutral-format assistant message for a response, to append to history."""
    content: list[dict[str, Any]] = []
    if resp.text:
        content.append({"type": "text", "text": resp.text})
    for call in resp.tool_calls:
        content.append(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        )
    return {"role": "assistant", "content": content}


def tool_result_message(results: list[tuple[str, str, bool]]) -> dict[str, Any]:
    """A user message of tool results: (tool_use_id, content, is_error) triples."""
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": i, "content": c, "is_error": err}
            for i, c, err in results
        ],
    }


def _retry_after(headers: Any) -> float | None:
    """The retry-after header in seconds. The HTTP-date form is not used by this API,
    and falls back to the normal backoff."""
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


class AnthropicProvider:
    """Calls the Anthropic Messages API. Construction needs a key; replay never builds one."""

    def __init__(self, api_key: str | None = None, max_tokens: int = DEFAULT_MAX_TOKENS):
        import anthropic  # lazy: replay must work without a key or a client

        self._anthropic = anthropic
        # Retries are owned by ModelClient, so the SDK's own retries are off.
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=0)
        self._max_tokens = max_tokens

    def complete(self, req: ModelRequest) -> ModelResponse:
        a = self._anthropic
        if req.model not in THINKING_BY_MODEL:
            raise ValueError(f"no thinking setting for model {req.model!r}")
        system = req.system
        kwargs: dict[str, Any] = {}
        if req.tools:
            kwargs["tools"] = req.tools
            if req.tool_choice is not None:
                # Forced tool_choice is rejected on current models, so steer from the prompt.
                system = f"{system}\n\nYou must respond by calling the `{req.tool_choice}` tool."
        try:
            msg = self._client.messages.create(
                model=req.model,
                max_tokens=self._max_tokens,
                system=system,
                messages=req.messages,
                # The pinned SDK's thinking parameter predates between_tools.
                extra_body={"thinking": THINKING_BY_MODEL[req.model]},
                **kwargs,
            )
        except a.APIConnectionError as e:
            raise TransientProviderError(str(e)) from e
        except a.APIStatusError as e:
            if e.status_code == 429 or e.status_code >= 500:
                raise TransientProviderError(
                    str(e), retry_after=_retry_after(e.response.headers)) from e
            raise
        text = "".join(b.text for b in msg.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in msg.content
            if b.type == "tool_use"
        ]
        return ModelResponse(
            text=text,
            tool_calls=calls,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            stop_reason=msg.stop_reason,
        )


class _NoProvider:
    def complete(self, req: ModelRequest) -> ModelResponse:
        raise RuntimeError("no provider configured; this client can only replay")


class ModelClient:
    def __init__(
        self,
        provider: Provider,
        mode: Mode,
        recordings: Path,
        namespace: str = "default",
        sleep: Callable[[float], None] = time.sleep,
    ):
        if mode not in MODES:
            raise ValueError(f"unknown LLM mode {mode!r}; use replay, record or live")
        self.provider = provider
        self.mode = mode
        self.recordings = Path(recordings)
        self.namespace = namespace
        self._sleep = sleep
        self.input_tokens = 0
        self.output_tokens = 0
        # (input, output) per request, since some prices depend on a request's length.
        self.calls: list[tuple[int, int]] = []

    @property
    def tokens_used(self) -> int:
        return self.input_tokens + self.output_tokens

    def complete(self, req: ModelRequest) -> ModelResponse:
        reuse = self.mode == "replay" or (
            self.mode == "record" and recording_exists(self.recordings, self.namespace, req))
        if reuse:  # record mode resumes: what is already paid for is not bought again
            resp = _response_from_dict(load_recording(self.recordings, self.namespace, req))
        else:
            resp = self._call_with_retry(req)
            if self.mode == "record":
                save_recording(self.recordings, self.namespace, req, resp)
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self.calls.append((resp.input_tokens, resp.output_tokens))
        return resp

    def _call_with_retry(self, req: ModelRequest) -> ModelResponse:
        for attempt in range(API_MAX_RETRIES + 1):
            try:
                return self.provider.complete(req)
            except TransientProviderError as e:
                if attempt == API_MAX_RETRIES:
                    raise
                wait = max(BACKOFF_BASE_SECONDS * 2**attempt, e.retry_after or 0)
                self._sleep(min(wait, MAX_RETRY_WAIT_SECONDS))
        raise AssertionError("unreachable")


def _response_from_dict(d: dict[str, Any]) -> ModelResponse:
    return ModelResponse(
        text=d["text"],
        tool_calls=[ToolCall(**c) for c in d["tool_calls"]],
        input_tokens=d["input_tokens"],
        output_tokens=d["output_tokens"],
        stop_reason=d["stop_reason"],
    )


def recordings_dir() -> Path:
    env = os.environ.get("LEDGER_RECORDINGS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "recordings"


def client_from_env(namespace: str) -> ModelClient:
    mode = os.environ.get("LEDGER_LLM_MODE", "replay")
    if mode not in MODES:
        raise ValueError(f"unknown LEDGER_LLM_MODE {mode!r}; use replay, record or live")
    provider = _NoProvider() if mode == "replay" else AnthropicProvider()
    return ModelClient(provider, mode, recordings_dir(), namespace=namespace)  # type: ignore[arg-type]
