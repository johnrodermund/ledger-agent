"""The audit graph: explore, hunt, confirm, then propose and verify each finding.

Nothing is applied to the database. There is no checkpointer, so a run keeps all
its state in memory and a rerun starts from the beginning.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from ledger_agent.agent.nodes import confirm, explore, hunt, pick, propose, verify
from ledger_agent.agent.state import AgentState, AuditResult, Deps, TraceEvent

# Each finding takes one pick and up to two nodes per fix attempt, so 40 findings at
# 3 attempts is 280 steps. This leaves a wide margin.
RECURSION_LIMIT = 2000

Node = Callable[[AgentState, Deps], dict[str, Any]]
__all__ = ["AuditResult", "TraceEvent", "audit", "build_graph"]


def _traced(name: str, node: Node, deps: Deps) -> Callable[[AgentState], dict[str, Any]]:
    def run(state: AgentState) -> dict[str, Any]:
        before = deps.client.tokens_used
        update = node(state, deps)
        in_fix = name in ("propose", "verify")
        deps.trace.append(TraceEvent(
            node=name,
            tokens=deps.client.tokens_used - before,
            queries=update.get("queries", state.get("queries", 0)) - state.get("queries", 0),
            attempt=state["attempt"] + 1 if in_fix else 0,
            finding_id=state["current"] if in_fix else None,
        ))
        return update

    return run


def _after_pick(state: AgentState) -> str:
    return END if state["current"] is None else "propose"


def _after_propose(state: AgentState) -> str:
    return "pick" if state["current"] is None else "verify"


def _after_verify(state: AgentState) -> str:
    return "pick" if state["current"] is None else "propose"


def build_graph(deps: Deps):
    graph = StateGraph(AgentState)
    nodes: dict[str, Node] = {
        "explore": explore, "hunt": hunt, "confirm": confirm,
        "pick": pick, "propose": propose, "verify": verify,
    }
    for name, node in nodes.items():
        graph.add_node(name, _traced(name, node, deps))
    graph.set_entry_point("explore")
    graph.add_edge("explore", "hunt")
    graph.add_edge("hunt", "confirm")
    graph.add_edge("confirm", "pick")
    graph.add_conditional_edges("pick", _after_pick, {END: END, "propose": "propose"})
    graph.add_conditional_edges("propose", _after_propose, {"pick": "pick", "verify": "verify"})
    graph.add_conditional_edges("verify", _after_verify, {"pick": "pick", "propose": "propose"})
    return graph.compile()


def initial_state(dbname: str) -> AgentState:
    return {
        "dbname": dbname, "schema_notes": "", "findings": [], "dropped": [], "queue": [],
        "current": None, "attempt": 0, "last_failure": None, "outcomes": [],
        "limits_hit": [], "evidence": {}, "proposal": None, "rejected": [], "malformed": [],
        "queries": 0, "propose_base": 0, "tool_results": [],
    }


def audit(dbname: str, deps: Deps) -> AuditResult:
    """Run the whole audit against dbname. Token totals count this run only."""
    deps.trace.clear()
    input_before = deps.client.input_tokens
    output_before = deps.client.output_tokens
    calls_before = len(deps.client.calls)
    final = build_graph(deps).invoke(
        initial_state(dbname), config={"recursion_limit": RECURSION_LIMIT})
    input_tokens = deps.client.input_tokens - input_before
    output_tokens = deps.client.output_tokens - output_before
    return AuditResult(
        state=final,
        trace=list(deps.trace),
        tokens=input_tokens + output_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        queries=final["queries"],
        calls=deps.client.calls[calls_before:],
    )
