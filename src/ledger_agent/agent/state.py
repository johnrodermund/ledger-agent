"""Agent state, limits and the records the nodes pass between them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel
from typing_extensions import TypedDict

from ledger_agent.llm.client import ModelClient
from ledger_agent.models import Correction, Finding


@dataclass(frozen=True)
class Limits:
    explore_queries: int = 15
    hunt_queries: int = 40
    fix_attempts: int = 3
    tokens: int = 200_000

    # Stage budgets are shares of tokens, each measured from the start of its stage.
    @property
    def explore_tokens(self) -> int:
        return self.tokens * 30 // 100

    @property
    def hunt_tokens(self) -> int:
        return self.tokens * 50 // 100

    @property
    def propose_tokens(self) -> int:
        return self.tokens * 20 // 100


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
    tool_results: list[str]  # the text of every query result shown to the model


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
