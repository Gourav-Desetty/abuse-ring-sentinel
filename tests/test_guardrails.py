import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from src.agent.graph_state import build_graph, run_agent
from src.agent.guardrails import SCORE_FLOOR, Guardrails, ValidationResult
from src.agent.nodes import FLAG_SCORE_FLOOR, RUN_FAILED_VERDICT


class GuardrailTests(unittest.TestCase):
    def setUp(self):
        self.llm = Mock()
        self.llm.invoke.return_value = SimpleNamespace(content="1. Yes\n2. C1 pays into C4.")
        self.verifier = self.llm.with_structured_output.return_value
        self.verifier.invoke.return_value = ValidationResult(is_grounded=True)
        self.guardrails = Guardrails(self.llm)

    def run_candidate(self, score):
        candidate = {
            "candidate_id": "shared_payout::C4",
            "method": "shared_payout",
            "members": ["C1", "C2", "C3"],
            "evidence": {"payout_account": "C4", "size": 3, "step_window": 3},
            "suspicion_score": score,
        }
        return run_agent(build_graph(self.llm, self.guardrails), candidate, thread_id="guardrail-test")

    def test_input_floor_is_inclusive_and_rejection_skips_model_calls(self):
        below = {"suspicion_score": SCORE_FLOOR - 0.01}
        at_floor = {"suspicion_score": SCORE_FLOOR}
        artifact = self.guardrails.input_guardrail([below, at_floor])

        self.assertEqual(artifact.passed, [at_floor])
        self.assertEqual(artifact.rejected, [below])
        self.assertEqual(self.run_candidate(below["suspicion_score"])["final_verdict"], "needs_more_data")
        self.llm.invoke.assert_not_called()
        self.verifier.invoke.assert_not_called()

    def test_grounded_yes_requires_payout_flag_floor(self):
        floor = FLAG_SCORE_FLOOR["shared_payout"]
        weak = self.run_candidate(floor - 0.01)
        strong = self.run_candidate(floor)

        self.assertTrue(weak["is_grounded"])
        self.assertEqual(weak["final_verdict"], "needs_more_data")
        self.assertEqual(strong["final_verdict"], "ring_flagged")
        self.llm.with_structured_output.assert_called_once_with(ValidationResult, method="json_schema")
        prompt = self.verifier.invoke.call_args.args[0]
        self.assertIn(json.dumps(["C1", "C2", "C3"]), prompt)
        self.assertIn('"payout_account": "C4"', prompt)

    def test_unsupported_claim_and_verifier_failure_have_distinct_outcomes(self):
        self.verifier.invoke.return_value = ValidationResult(
            is_grounded=False, hallucinated_claims=["Unknown payer C9"]
        )
        rejected = self.run_candidate(0.9)

        self.assertEqual(rejected["final_verdict"], "needs_more_data")
        self.assertEqual(rejected["hallucinated_claims"], ["Unknown payer C9"])
        self.assertFalse(rejected["run_failed"])

        self.verifier.invoke.side_effect = ValueError("invalid structured response")
        failed = self.run_candidate(0.9)

        self.assertFalse(failed["is_grounded"])
        self.assertTrue(failed["run_failed"])
        self.assertEqual(failed["failure_stage"], "verify")
        self.assertEqual(failed["final_verdict"], RUN_FAILED_VERDICT)
