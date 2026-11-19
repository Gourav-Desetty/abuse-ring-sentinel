import unittest
from dataclasses import replace
from unittest.mock import patch

from src.agent.nodes import RUN_FAILED_VERDICT
from src.detection.detection import Candidate
from src.eval import metrics


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.candidate = Candidate("candidate-1", "shared_payout", ["C1"], {}, 0.4)

    def test_labels_train_matches_test_hard_negatives_and_untagged_candidates(self):
        tag = {"ring_id": "R1", "ring_type": "shared_payout", "difficulty": "obvious", "split": "train"}
        cases = [
            ({"C1": tag}, "positive", False),
            ({"C1": {**tag, "split": "test"}}, "positive", True),
            ({"C1": {**tag, "ring_type": "legit_cluster", "split": "test"}}, "negative", True),
            ({}, "negative", True),
        ]
        for tags, ground_truth, eligible in cases:
            with self.subTest(tags=tags):
                label = metrics.label_candidate(self.candidate, tags)
                self.assertEqual(label["ground_truth"], ground_truth)
                self.assertEqual(label["eligible"], eligible)

    def test_repeated_runs_exclude_failures_from_the_vote_and_flip_rate(self):
        outcomes = [
            {"final_verdict": "ring_flagged"},
            RuntimeError("quota exhausted"),
            {"final_verdict": "cleared"},
            {"final_verdict": "ring_flagged"},
        ]
        with patch.object(metrics, "run_agent", side_effect=outcomes) as run_agent:
            run = metrics.run_candidate_repeated(None, self.candidate, "eval", n_runs=4)

        self.assertEqual(run.modal_verdict, "ring_flagged")
        self.assertAlmostEqual(run.flip_rate, 1 / 3)
        self.assertEqual(run.verdicts, ["ring_flagged", RUN_FAILED_VERDICT, "cleared", "ring_flagged"])
        self.assertEqual((run.n_succeeded, run.n_failed), (3, 1))
        self.assertEqual(run.failure_errors, ["RuntimeError: quota exhausted"])
        self.assertEqual(
            [call.kwargs["thread_id"] for call in run_agent.call_args_list],
            ["eval-0", "eval-1", "eval-2", "eval-3"],
        )

    def test_candidates_with_no_verdict_are_excluded_from_scoring(self):
        with patch.object(metrics, "run_agent", side_effect=RuntimeError("quota exhausted")):
            run = metrics.run_candidate_repeated(None, self.candidate, "eval", n_runs=2)

        self.assertEqual(run.modal_verdict, metrics.NO_VERDICT)
        self.assertIsNone(run.flip_rate)

        record = metrics.CandidateRecord(
            candidate_id="candidate-1", method="shared_payout", suspicion_score=0.4, members=["C1"],
            evidence={}, ring_id="R1", ring_type="shared_payout", difficulty="obvious", split="test",
            ground_truth="positive", predicted_verdict=run.modal_verdict, n_runs=2,
            verdicts=run.verdicts, flip_rate=run.flip_rate, is_grounded=None, hallucinated_claims=[],
            diagnosis=None, reason=None,
        )
        flagged = replace(record, candidate_id="candidate-2", predicted_verdict="ring_flagged")
        scored = metrics.compute_metrics([record, flagged])

        self.assertEqual((scored.tp, scored.fn, scored.recall), (1, 0, 1.0))
        row, = metrics.compute_breakdown([record, flagged])
        self.assertEqual((row.n, row.n_flagged), (1, 1))
