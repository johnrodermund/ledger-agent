"""Agent state, limits and the records the nodes pass between them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

from pydantic import BaseModel

from ledger_agent.llm.client import ModelClient
from ledger_agent.models import Correction, Finding


@dataclass(frozen=True)
class Limits:
    explore_queries: int = 15
    hunt_queries: int = 40
    fix_attempts: int = 3
    tokens: int = 200_000


class DroppedFinding(BaseModel):
    """A submitted finding that did not survive checking, and why."""

    finding_id: str | None
    summary: str
    reason: str


class Outcome(BaseModel):
    finding_id: str
    status: Literal["verified", "needs_human"]
    correction: Correction | None = None
    attempts: int
    reason: str | None = None


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


@dataclass
class Deps:
    client: ModelClient
    limits: Limits = field(default_factory=Limits)
