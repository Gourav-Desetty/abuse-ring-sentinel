"""Agent nodes for detection, diagnosis, independent verification, and verdicts."""

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

# Evaluation excludes runs that fail to produce a judgment.
RUN_FAILED_VERDICT = "run_failed"

# Avoid a circular import with graph_state.py. LangGraph resolves router hints at runtime.
State = dict[str, Any]

logger = logging.getLogger(__name__)

MODEL_NAME = "openai/gpt-oss-120b"


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
    """Apply the input score floor even when callers invoke the graph directly."""
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
    return "failed" if state.get("run_failed") else "continue"


def verify_node(state: State, *, guardrails: Guardrails) -> dict[str, Any]:
    candidate = state["candidate"]
    diagnosis = state["diagnosis"]
    assert diagnosis is not None  # verify only runs after diagnose on the "continue" path
    # Method and member IDs can be cited because the diagnosis prompt includes them.
    grounding_source = {"method": candidate["method"], "members": candidate["members"], **candidate["evidence"]}
    result = guardrails.output_guardrail(diagnosis=diagnosis, evidence=grounding_source)
    update: dict[str, Any] = {"is_grounded": result.is_grounded, "hallucinated_claims": result.hallucinated_claims}
    if result.llm_failed:
        update |= {"run_failed": True, "failure_stage": "verify", "failure_error": result.failure_error}
    return update


_Q1_ANSWER_RE = re.compile(r"1\.\s*.*?\b(Yes|No|Unclear)\b", re.IGNORECASE | re.DOTALL)

# Coincidental payouts can look grounded because they share timing and size signals
# with planted rings. The flag floor favors precision at the cost of recall.
FLAG_SCORE_FLOOR = {"shared_payout": 0.4}


def decide_node(state: State) -> dict[str, Any]:
    """Resolve a verdict from grounding, the numbered Q1 answer, and the flag floor."""
    candidate = state["candidate"]
    # A failed verifier also sets is_grounded=False, so check run failure first.
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
