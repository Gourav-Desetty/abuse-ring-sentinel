"""Run ring detection, diagnosis, verification, and verdicts in a checkpointed LangGraph workflow."""

from __future__ import annotations

import functools
from typing import Any, TypedDict

from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph

from src.agent.guardrails import CandidateDict, Guardrails
from src.agent.nodes import decide_node, detect_node, diagnose_node, route_after_detect, route_after_diagnose, verify_node


class RingState(TypedDict):
    candidate: CandidateDict
    input_guardrail_passed: bool | None
    diagnosis: str | None
    is_grounded: bool | None
    hallucinated_claims: list[str]
    final_verdict: str | None
    reason: str | None
    # Infrastructure-failure trail, distinct from any verdict the agent
    # could reach. run_failed=True means one of the LLM calls never
    # returned an answer (retries exhausted), so this run produced no
    # judgment at all -- see nodes.RUN_FAILED_VERDICT.
    run_failed: bool
    failure_stage: str | None  # "diagnose" | "verify"
    failure_error: str | None


def initial_state(candidate: CandidateDict) -> RingState:
    return RingState(
        candidate=candidate,
        input_guardrail_passed=None,
        diagnosis=None,
        is_grounded=None,
        hallucinated_claims=[],
        final_verdict=None,
        reason=None,
        run_failed=False,
        failure_stage=None,
        failure_error=None,
    )


def build_graph(
    llm: ChatGroq, guardrails: Guardrails, checkpointer: InMemorySaver | None = None,
    *, flag_score_floor: dict[str, float] | None = None,
):
    """Compile a reusable agent graph that pauses before the verdict for review."""
    graph = StateGraph(RingState)
    graph.add_node("detect", functools.partial(detect_node, guardrails=guardrails))
    graph.add_node("diagnose", functools.partial(diagnose_node, llm=llm))
    graph.add_node("verify", functools.partial(verify_node, guardrails=guardrails))
    graph.add_node("decide", functools.partial(decide_node, flag_score_floor=flag_score_floor))

    graph.add_edge(START, "detect")
    graph.add_conditional_edges("detect", route_after_detect, {"continue": "diagnose", "reject": END})
    graph.add_conditional_edges("diagnose", route_after_diagnose, {"continue": "verify", "failed": END})
    graph.add_edge("verify", "decide")
    graph.add_edge("decide", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver(), interrupt_before=["decide"])


def run_agent(app, candidate: CandidateDict, thread_id: str, auto_resume: bool = True) -> dict[str, Any]:
    """Run a candidate and return its state, resuming past the review checkpoint when auto_resume is enabled."""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(initial_state(candidate), config)
    if auto_resume and app.get_state(config).next:
        result = app.invoke(None, config)
    return result


def resume(app, thread_id: str) -> dict[str, Any]:
    """Resume a reviewed thread on the graph that created its checkpoint and return the final state."""
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(None, config)
