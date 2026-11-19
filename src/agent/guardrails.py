"""Filter low-scoring candidates and verify concrete claims against graph evidence."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from src.agent.retry import invoke_with_retry

logger = logging.getLogger(__name__)

CandidateDict = dict[str, Any]

# Catch degenerate scores after detection's graph filters.
SCORE_FLOOR = 0.05


class ValidationResult(BaseModel):
    is_grounded: bool = Field(description="True if every claim in the diagnosis is supported by the graph evidence")
    hallucinated_claims: list[str] = Field(default_factory=list, description="Specific claims not grounded in the graph evidence")


@dataclass
class InputGuardrailArtifact:
    passed: list[CandidateDict]
    rejected: list[CandidateDict]


@dataclass
class OutputGuardrailArtifact:
    is_grounded: bool
    hallucinated_claims: list[str] = field(default_factory=list)
    # True when the verifier failed to return a judgment.
    llm_failed: bool = False
    failure_error: str | None = None


_VALIDATION_PROMPT = """
    You are a fact checker for fraud-ring analyses. Your job is narrow: catch INVENTED FACTS,
    not opinions.

    DIAGNOSIS:
    {diagnosis}

    GRAPH EVIDENCE (source of truth):
    {evidence}

    Check ONLY concrete, checkable facts in the diagnosis: specific numbers, account ids,
    device ids, cycle lengths, amounts, or step/time-windows. Each such fact must be
    consistent with the graph evidence above (present in it, or a straightforward
    restatement of it -- e.g. repeating a step_window value or listing member account ids the
    evidence already names is NOT a hallucination).

    Do NOT flag the diagnosis's own interpretive conclusions as hallucinations: its
    Yes/No/Unclear judgment, its named collusion mechanism (e.g. "shared-device", "payout
    funnel"), and its severity rating (Low/Medium/High/Critical) are opinions the diagnosis is
    entitled to form from the evidence, not factual claims that must appear in the evidence
    verbatim. Only flag a hallucination when the diagnosis states a specific number, id, or
    fact that is absent from or contradicts the graph evidence.

    Reply ONLY in this JSON format with no extra text:
    {{"is_grounded": true, "hallucinated_claims": []}}
    """


class Guardrails:
    def __init__(self, llm) -> None:
        self.llm = llm
        # Groq's tool-calling path failed for this model. See docs/EVALUATION.md.
        self.structured_llm = llm.with_structured_output(ValidationResult, method="json_schema")

    def input_guardrail(self, candidates: list[CandidateDict], score_floor: float = SCORE_FLOOR) -> InputGuardrailArtifact:
        passed = [c for c in candidates if c["suspicion_score"] >= score_floor]
        rejected = [c for c in candidates if c["suspicion_score"] < score_floor]
        if rejected:
            logger.info(f"input guardrail rejected {len(rejected)} candidate(s) below score floor {score_floor}")
        return InputGuardrailArtifact(passed=passed, rejected=rejected)

    def output_guardrail(self, diagnosis: str, evidence: dict[str, Any]) -> OutputGuardrailArtifact:
        validation_input = _VALIDATION_PROMPT.format(diagnosis=diagnosis, evidence=json.dumps(evidence, default=str))

        try:
            validation_data: ValidationResult = invoke_with_retry(lambda: self.structured_llm.invoke(validation_input))
        except Exception as parse_err:
            # Fail closed and preserve the missing judgment for evaluation.
            logger.error(f"output guardrail LLM call failed: {parse_err}")
            return OutputGuardrailArtifact(
                is_grounded=False,
                hallucinated_claims=["Guardrail could not obtain an LLM validation response."],
                llm_failed=True,
                failure_error=f"{type(parse_err).__name__}: {parse_err}",
            )

        is_grounded = validation_data.is_grounded
        hallucinations = validation_data.hallucinated_claims

        if is_grounded:
            logger.info("diagnosis grounded in graph evidence")
        else:
            logger.warning(f"hallucinations detected: {hallucinations}")

        return OutputGuardrailArtifact(is_grounded=is_grounded, hallucinated_claims=hallucinations)
