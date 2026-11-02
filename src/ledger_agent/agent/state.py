"""Agent state, limits and the records the nodes pass between them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from pydantic import BaseModel

from ledger_agent.llm.client import MODEL, ModelClient
from ledger_agent.models import Correction, Finding


# Each model's limit for a whole run. Sonnet writes longer notes and ran out of
# Haiku's explore budget; Haiku keeps the limit its recordings were made under.
TOKEN_LIMITS = {"claude-haiku-5-5": 300_000, "claude-sonnet-5-5": 600_000}


@dataclass(frozen=True)
class Limits:
    explore_queries: int = 15
    hunt_queries: int = 40
    fix_attempts: int = 3
    tokens: int = field(default_factory=lambda: TOKEN_LIMITS[MODEL])

    # Stage budgets are shares of tokens, each measured from the start of its stage.
    @property
    def explore_tokens(self) -> int:
        return self.tokens * 25 // 100

    @property
    def hunt_tokens(self) -> int:
        return self.tokens * 35 // 100

    @property
    def propose_tokens(self) -> int:
        return self.tokens * 40 // 100


class DroppedFinding(BaseModel):
    """A submitted finding that did not survive checking, and why."""

    finding_id: str | None
    summary: str
    reason: str


class Outcome(BaseModel):
    finding_id: str
    status: Literal["verified", "needs_human", "refused"]
    correction: Correction | None = None
    attempts: int
    reason: str | None = None
    rendered_sql: str | None = None


class Evidence(BaseModel):
    """The rows a finding's evidence query returned when it was confirmed. Values are
    the strings the query tool produced, so they serialise exactly."""

    columns: list[str]
    rows: list[list[str]]
    truncated: bool = False


class TraceEvent(BaseModel):
    node: str
    tokens: int
    queries: int
    attempt: int
    finding_id: str | None = None


class AgentState(TypedDict):
    dbname: str
    schema_notes: str
    findings: list[Finding]
    dropped: list[DroppedFinding]
    queue: list[str]
    current: str | None
    attempt: int
    last_failure: str | None
    outcomes: list[Outcome]
    limits_hit: list[str]
    evidence: dict[str, Evidence]
    proposal: Correction | None
    rejected: list[Correction]  # every correction the verifier refused, in order
    malformed: list[dict[str, Any]]  # raw arguments of submissions that failed validation
    queries: int
    propose_base: int  # the client's token total when propose began
    tool_results: list[str]  # query results and propose prompts the model was sent


@dataclass
class Deps:
    client: ModelClient
    limits: Limits = field(default_factory=Limits)
    trace: list[TraceEvent] = field(default_factory=list)


class AuditResult(BaseModel):
    """What one audit produced. Decimal amounts serialise as strings."""

    state: AgentState
    trace: list[TraceEvent]
    tokens: int
    input_tokens: int
    output_tokens: int
    queries: int
    # (input, output, cache_write, cache_read) tokens per model request; input is uncached
    calls: list[tuple[int, int, int, int]] = []
