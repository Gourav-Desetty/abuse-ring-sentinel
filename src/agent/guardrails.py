"""Dual guardrail layer: a score floor before the LLM, a grounding check
after it.

input_guardrail()'s floor is deliberately low (0.05) -- detection.py
already hard-filters candidates before they arrive here, so this exists to
catch degenerate scores, not to re-filter its work.

output_guardrail() fact-checks a diagnosis against the candidate's own
graph evidence via a second, structured-output LLM call, scoped to invented
facts only (numbers/ids/amounts absent from the evidence) -- not the
diagnosis's own interpretive judgment, since an earlier version that
flagged phrases like "severity: Medium" as unsupported claims cost real
recall. A failed LLM call still fails safe to `is_grounded=False`, but sets
`llm_failed=True` so that's never mistaken downstream for a genuine
"not grounded" verdict.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from src.agent.retry import invoke_with_retry

logger = logging.getLogger(__name__)

CandidateDict = dict[str, Any]

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
    # True when the guardrail never actually got an answer out of the LLM
    # (exhausted retries / unparseable response). `is_grounded=False` is
    # still set as the fail-safe, but this flag is what tells downstream
    # code the False is an infrastructure artifact, not a judgment.
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
        # Binds the LLM to a schema so it returns structured data directly
        # instead of freeform text we have to parse. The default method
        # ("function_calling"/tool-calling) 400s against openai/gpt-oss-120b
        # on Groq ("Tool choice is required, but model did not call a
        # tool") -- "json_schema" is what actually works for this pairing.
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
            # Fail-safe, not fail-hard: if the LLM's structured output call
            # itself errors, don't crash the pipeline. Mark as ungrounded --
            # the guardrail is a safety check, so a failure should deny by
            # default, not silently pass -- but ALSO set llm_failed, because
            # this False means "no answer" and must never be mistaken (in
            # the saved eval data) for the LLM having judged the diagnosis
            # ungrounded. Every exception here is a failure to obtain a
            # judgment, so all of them set the flag, not just LLMCallFailed.
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
