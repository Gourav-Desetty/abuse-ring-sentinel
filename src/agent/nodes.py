"""The four node functions: detect -> diagnose -> verify -> decide.

diagnose_node and verify_node split verdict generation from fact-checking
into two separate model calls -- a freeform `.invoke()` produces a diagnosis
citing evidence, then a separate structured-output call (`Guardrails.output_guardrail`)
checks it independently, rather than trusting one call's self-reported
citations.

`RING_ANALYSIS_PROMPT` ends in a numbered Yes/No/Unclear judgment;
`decide_node` parses that answer directly rather than free-text sentiment.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from langchain_groq import ChatGroq
from pydantic import SecretStr

from src.agent.guardrails import CandidateDict, Guardrails
from src.agent.retry import LLMCallFailed, invoke_with_retry

# Sentinel `final_verdict` for a run that never produced an LLM judgment at
# all (exhausted retries against Groq). Deliberately NOT one of the three
# real verdicts: an infrastructure failure must not be scoreable as if the
# agent had declined to flag.
RUN_FAILED_VERDICT = "run_failed"

# Node functions take the plain dict shape of graph_state.RingState rather
# than importing that TypedDict directly: graph_state.py imports these node
# functions to wire the graph, so importing RingState back here would be
# circular. It's also required for correctness, not just style -- LangGraph
# resolves add_conditional_edges' router function's type hints at runtime
# (get_type_hints), and a string-quoted forward ref to a name never actually
# imported into this module's namespace fails that resolution.
State = dict[str, Any]

logger = logging.getLogger(__name__)

MODEL_NAME = "openai/gpt-oss-120b"  # replacement for llama-3.3-70b-versatile, deprecated by Groq


@dataclass
class GroqConfig:
    api_key: str
    model: str = MODEL_NAME
    temperature: float = 0.1  # low, for more consistent verdicts

    @classmethod
    def from_env(cls) -> "GroqConfig":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY must be set (environment or .env) to run the agent.")
        return cls(api_key=api_key)


def build_llm(config: GroqConfig | None = None) -> ChatGroq:
    config = config or GroqConfig.from_env()
    return ChatGroq(api_key=SecretStr(config.api_key), model=config.model, temperature=config.temperature)


RING_ANALYSIS_PROMPT = """You are a fraud-ring analyst.
Use ONLY the graph evidence below. Do not invent account ids, amounts, device ids, cycle lengths, or step/time windows.
If a detail is not explicitly supported by the evidence, say "Unclear".

CANDIDATE RING
Detection method: {method}
Members ({n_members}): {members}

GRAPH EVIDENCE:
{evidence}

Return a concise analysis with:
1. Is this likely a collusion ring? (Yes/No/Unclear)
2. Which specific evidence supports that? Cite exact fields/values from the graph evidence above.
3. What is the plausible collusion mechanism (shared device, payout funnel, circular flow)?
4. Overall suspicion severity? (Low/Medium/High/Critical)

For every specific claim, cite the exact supporting evidence field. If the evidence is too thin to
support a real judgment (e.g. a single coincidental fan-in with no corroborating signal), answer
"Unclear" rather than guessing."""


def detect_node(state: State, *, guardrails: Guardrails) -> dict[str, Any]:
    """Entry node. A batch pipeline should pre-filter with
    `Guardrails.input_guardrail()` before invoking the graph per-candidate;
    this node re-applies the same score-floor check defensively, so a graph
    invoked directly on any one candidate (as in a smoke test) is still safe
    on its own -- not a second, stricter filter, just the same one again.
    """
    candidate: CandidateDict = state["candidate"]
    result = guardrails.input_guardrail([candidate])
    passed = bool(result.passed)
    return {
        "input_guardrail_passed": passed,
        "final_verdict": None if passed else "needs_more_data",
        "reason": None if passed else f"input guardrail: suspicion_score {candidate['suspicion_score']} below floor",
    }


def route_after_detect(state: State) -> str:
    return "continue" if state["input_guardrail_passed"] else "reject"


def diagnose_node(state: State, *, llm: ChatGroq) -> dict[str, Any]:
    candidate = state["candidate"]
    prompt = RING_ANALYSIS_PROMPT.format(
        method=candidate["method"],
        n_members=len(candidate["members"]),
        members=", ".join(candidate["members"]),
        evidence=json.dumps(candidate["evidence"], indent=2, default=str),
    )
    try:
        response = invoke_with_retry(lambda: llm.invoke(prompt))
    except LLMCallFailed as err:
        # A Groq outage (typically the daily token quota) must not take the
        # whole batch down with it, and must not be recorded as a verdict.
        # Short-circuit to END via route_after_diagnose with the run marked
        # failed; the eval layer drops it from the majority vote.
        logger.error(f"diagnose LLM call failed for {candidate['candidate_id']}: {err}")
        return {
            "diagnosis": None,
            "run_failed": True,
            "failure_stage": "diagnose",
            "failure_error": f"{type(err).__name__}: {err}",
            "final_verdict": RUN_FAILED_VERDICT,
            "reason": f"LLM call failed during diagnose (no judgment produced): {err}",
        }
    return {"diagnosis": str(response.content)}


def route_after_diagnose(state: State) -> str:
    """Skip verify/decide entirely when diagnose never produced a diagnosis
    -- there is nothing to fact-check or decide on, and burning a second
    LLM call on a provider that just refused one is pointless.
    """
    return "failed" if state.get("run_failed") else "continue"


def verify_node(state: State, *, guardrails: Guardrails) -> dict[str, Any]:
    candidate = state["candidate"]
    diagnosis = state["diagnosis"]
    assert diagnosis is not None  # verify only runs after diagnose on the "continue" path
    # Ground against everything diagnose_node's prompt actually supplied --
    # method + members + evidence -- not evidence alone. diagnose_node's
    # prompt legitimately includes the member account ids as context; if the
    # grounding check only saw `evidence`, citing those same ids back would
    # look like a hallucination even though nothing was invented.
    grounding_source = {"method": candidate["method"], "members": candidate["members"], **candidate["evidence"]}
    result = guardrails.output_guardrail(diagnosis=diagnosis, evidence=grounding_source)
    update: dict[str, Any] = {"is_grounded": result.is_grounded, "hallucinated_claims": result.hallucinated_claims}
    if result.llm_failed:
        # is_grounded is False here only as a fail-safe default -- carry the
        # fact that no judgment was obtained through to decide_node, which
        # would otherwise read it as a genuine "not grounded".
        update |= {"run_failed": True, "failure_stage": "verify", "failure_error": result.failure_error}
    return update


_Q1_ANSWER_RE = re.compile(r"1\.\s*.*?\b(Yes|No|Unclear)\b", re.IGNORECASE | re.DOTALL)

# The governing rule is: never flag on weak evidence. Grounding alone
# doesn't enforce that -- it only catches invented facts, not an honest but
# overconfident read of genuinely thin evidence. detection.py's own
# calibration (see its module docstring) found real, coincidental PaySim
# shared_payout fan-in scores in the *same* 0.07-0.5 range as planted
# rings, because nothing separates them but timing/size, both of which a
# diagnosis can legitimately cite without hallucinating anything. So for
# shared_payout specifically, a "Yes" + grounded diagnosis still isn't
# enough to flag below this score: it forces needs_more_data instead, even
# though nothing was ungrounded. This trades recall on subtle shared_payout
# rings for precision -- the right side of that trade for a flag with real
# consequences. shared_device and circular_flow aren't included: their
# evidence (an actual shared device_id; a closed, amount-consistent cycle)
# doesn't have this coincidental-collision problem, so grounded+"Yes" is
# already a meaningful bar for them.
FLAG_SCORE_FLOOR = {"shared_payout": 0.4}


def decide_node(state: State) -> dict[str, Any]:
    """Combine the output guardrail's grounding check, the shared_payout
    score floor, and the diagnosis's own Q1 answer into a final verdict.
    An ungrounded diagnosis, an ambiguous ("Unclear") one, or a "Yes" that
    doesn't clear its method's flag-score floor all resolve to
    "needs_more_data" -- never "ring_flagged" on weak evidence. A run whose
    LLM calls failed outright resolves to RUN_FAILED_VERDICT instead, which
    is not a verdict at all and is excluded from scoring downstream.
    """
    candidate = state["candidate"]
    # Checked before the grounding branch: when the output guardrail's LLM
    # call failed, is_grounded is False by fail-safe default, not by
    # judgment, and must not resolve to a normal "needs_more_data".
    if state.get("run_failed"):
        return {
            "final_verdict": RUN_FAILED_VERDICT,
            "reason": (
                f"LLM call failed during {state.get('failure_stage')} (no judgment produced): "
                f"{state.get('failure_error')}"
            ),
        }

    if not state.get("is_grounded", False):
        return {
            "final_verdict": "needs_more_data",
            "reason": f"output guardrail rejected diagnosis: unsupported claims {state.get('hallucinated_claims')}",
        }

    diagnosis = state["diagnosis"]
    assert diagnosis is not None  # decide only reaches here after diagnose/verify on the "continue" path
    match = _Q1_ANSWER_RE.search(diagnosis)
    answer = match.group(1).lower() if match else "unclear"

    score_floor = FLAG_SCORE_FLOOR.get(candidate["method"], 0.0)
    if answer == "yes" and candidate["suspicion_score"] < score_floor:
        return {
            "final_verdict": "needs_more_data",
            "reason": (
                f"diagnosis said 'yes' and was grounded, but suspicion_score {candidate['suspicion_score']} "
                f"is below the {candidate['method']} flag floor {score_floor} -- evidence too thin to act on"
            ),
        }

    if answer == "yes":
        final_verdict = "ring_flagged"
    elif answer == "no":
        final_verdict = "cleared"
    else:
        final_verdict = "needs_more_data"

    return {"final_verdict": final_verdict, "reason": f"diagnosis answered '{answer}' to Q1, grounded=True"}
