"""The LangGraph state machine: detect -> diagnose -> verify -> decide.

`detect` and `diagnose` can each short-circuit straight to END: `detect`
skips the rest when the candidate fails the input guardrail's score floor
(no LLM call wasted on something that shouldn't reach the agent); `diagnose`
skips the rest when its LLM call exhausts retries, ending the run with
`final_verdict=nodes.RUN_FAILED_VERDICT` -- deliberately not one of the
three real verdicts, so downstream scoring drops it rather than counting an
infrastructure outage as a declined flag.

Compiled with `interrupt_before=["decide"]` so execution pauses after
verify for a human to inspect the diagnosis before the verdict finalizes.
`run_agent()` auto-resumes by default (batch/eval usage); a review UI would
call `resume()` only after approval.
"""

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


def build_graph(llm: ChatGroq, guardrails: Guardrails, checkpointer: InMemorySaver | None = None):
    """Wire the four nodes into a compiled, checkpointed LangGraph app.

    Build this once per process/batch and reuse the returned app across
    candidates (via run_agent/resume) -- each compiled app owns one
    checkpointer, and a thread_id paused mid-graph can only be resumed
    against the same checkpointer that paused it, not a freshly-built one.
    """
    graph = StateGraph(RingState)
    graph.add_node("detect", functools.partial(detect_node, guardrails=guardrails))
    graph.add_node("diagnose", functools.partial(diagnose_node, llm=llm))
    graph.add_node("verify", functools.partial(verify_node, guardrails=guardrails))
    graph.add_node("decide", decide_node)

    graph.add_edge(START, "detect")
    graph.add_conditional_edges("detect", route_after_detect, {"continue": "diagnose", "reject": END})
    graph.add_conditional_edges("diagnose", route_after_diagnose, {"continue": "verify", "failed": END})
    graph.add_edge("verify", "decide")
    graph.add_edge("decide", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver(), interrupt_before=["decide"])


def run_agent(app, candidate: CandidateDict, thread_id: str, auto_resume: bool = True) -> dict[str, Any]:
    """Run one candidate through a compiled `app` (see build_graph). Returns
    the final state dict.

    If the input guardrail rejects the candidate, this returns immediately
    (final_verdict="needs_more_data") without pausing -- there's nothing
    for a human to review yet. Otherwise it runs detect -> diagnose ->
    verify, pauses before decide, and (if auto_resume) resumes right away.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(initial_state(candidate), config)
    if auto_resume and app.get_state(config).next:
        result = app.invoke(None, config)
    return result


def resume(app, thread_id: str) -> dict[str, Any]:
    """Resume a graph paused at the interrupt_before=["decide"] checkpoint
    -- call after a human has reviewed the diagnosis and approves finalizing
    the verdict. `app` must be the same compiled graph (same checkpointer)
    that paused this thread_id.
    """
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(None, config)
