"""Scripted model provider for tests. No test may reach the real API."""

import copy

from ledger_agent.llm.client import ModelRequest, ModelResponse, ToolCall


class FakeProvider:
    """Returns scripted responses in order and records every request.

    An empty ``ModelResponse(text="")`` simulates a forced submit that comes
    back without a tool call. Running past the end of the script raises.
    """

    def __init__(self, script: list[ModelResponse]):
        self.script = list(script)
        self.requests: list[ModelRequest] = []

    def complete(self, req: ModelRequest) -> ModelResponse:
        self.requests.append(copy.deepcopy(req))  # the caller keeps appending to messages
        if not self.script:
            raise AssertionError("FakeProvider script exhausted")
        return self.script.pop(0)


def call(name: str, call_id: str = "c1", **arguments) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def reply(*calls: ToolCall, text: str = "", tokens: int = 0) -> ModelResponse:
    return ModelResponse(text=text, tool_calls=list(calls), input_tokens=tokens)
