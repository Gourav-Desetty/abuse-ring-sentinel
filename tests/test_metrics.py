import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.agent.guardrails import ValidationResult
from src.agent.nodes import RUN_FAILED_VERDICT
from src.detection.detection import Candidate
from src.eval import metrics


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.candidate = Candidate("candidate-1", "shared_payout", ["C1"], {}, 0.4)

    def test_labels_require_full_same_method_match_and_ignore_member_order(self):
        candidate = replace(self.candidate, members=["C1", "C2", "C3"], evidence={"payout_account": "P"})
        tag = {
            "ring_id": "R1", "ring_type": "shared_payout", "difficulty": "obvious",
            "split": "test", "payout_account": "P",
        }
        accounts = ["C1", "C2", "C3", "P"]
        test_tags = dict.fromkeys(accounts, tag)
        cases = [
            (candidate, test_tags, "positive", True),
            (replace(candidate, members=list(reversed(candidate.members))), test_tags, "positive", True),
            (candidate, dict.fromkeys(accounts, {**tag, "split": "train"}), "positive", False),
            (candidate, dict.fromkeys(accounts, {**tag, "ring_type": "legit_personal_transfers"}), "negative", True),
            (replace(candidate, members=["C1", "C2", "X"]), test_tags, None, False),
            (replace(candidate, evidence={"payout_account": "X"}), test_tags, None, False),
            (replace(candidate, members=["P", "C2", "C3"], evidence={"payout_account": "C1"}),
             test_tags, None, False),
            (replace(candidate, method="shared_device"), test_tags, None, False),
            (replace(candidate, members=["X", *candidate.members]),
             {**test_tags, "X": {**tag, "ring_id": "R2", "split": "train"}}, None, False),
            (candidate, {"C1": tag}, None, False),
            (candidate, {}, None, False),
        ]
        for detected, tags, ground_truth, eligible in cases:
            with self.subTest(detected=detected, tags=tags):
                label = metrics.label_candidate(detected, tags)
                self.assertEqual(label["ground_truth"], ground_truth)
                self.assertEqual(label["eligible"], eligible)

    def test_flag_floors_use_only_calibration_and_one_candidate_per_cohort(self):
        rows = []
        for ring, score, split, truth in [
            ("p", 0.7, "train", "positive"), ("p", 0.55, "train", "positive"),
            ("n", 0.3, "train", "negative"), ("t1", 0.2, "test", "positive"),
            ("t2", 0.8, "test", "negative"),
        ]:
            candidate = replace(self.candidate, candidate_id=f"{ring}-{score}", suspicion_score=score)
            rows.append((candidate, {"ring_id": ring, "split": split, "ground_truth": truth}))
        one_class = replace(self.candidate, method="shared_device", suspicion_score=0.8)
        rows.append((one_class, {"ring_id": "device", "split": "train", "ground_truth": "positive"}))

        self.assertEqual(metrics.calibrate_flag_floors(rows), {
            "shared_payout": metrics.MethodCalibration(0.7, 1, 1),
            "shared_device": metrics.MethodCalibration(None, 1, 0),
            "circular_flow": metrics.MethodCalibration(None, 0, 0),
        })

    def test_evaluation_applies_each_methods_calibrated_floor(self):
        rows = []
        for method, floor in {"shared_device": 0.85, "circular_flow": 0.65, "shared_payout": 0.55}.items():
            for case, split, truth, score in [
                ("cal-positive", "train", "positive", floor),
                ("cal-negative", "train", "negative", 0.3),
                ("test-at-floor", "test", "positive", floor),
                ("test-below-floor", "test", "negative", 0.5),
            ]:
                ring_id = f"{method}-{case}"
                candidate = replace(self.candidate, candidate_id=ring_id, method=method, suspicion_score=score)
                label = {
                    "ring_id": ring_id, "ring_type": method, "difficulty": None,
                    "split": split, "ground_truth": truth,
                }
                rows.append((candidate, label))

        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(content="1. Yes")
        llm.with_structured_output.return_value.invoke.return_value = ValidationResult(is_grounded=True)
        with (
            patch.object(metrics.Neo4jConfig, "from_env"),
            patch.object(metrics.GroqConfig, "from_env"),
            patch.object(metrics, "get_neo4j_driver"),
            patch.object(metrics, "detect_candidates", return_value=[candidate for candidate, _ in rows]),
            patch.object(metrics, "fetch_ring_tags", return_value={}),
            patch.object(metrics, "label_candidates", return_value=rows),
            patch.object(metrics, "build_llm", return_value=llm),
        ):
            result = metrics.run_evaluation(n_main_runs=1, n_stability_runs=0)

        self.assertEqual(len(result.records), 6)
        for record in result.records:
            with self.subTest(method=record.method, truth=record.ground_truth):
                expected = "ring_flagged" if record.ground_truth == "positive" else "needs_more_data"
                self.assertEqual(record.predicted_verdict, expected)
                self.assertTrue(record.is_grounded)

    def test_ring_recall_counts_undetected_rings_and_deduplicates_detections(self):
        tags = {
            account: {"ring_id": group, "split": "test", "ring_type": "shared_payout"}
            for account, group in (("A", "found"), ("B", "found"), ("C", "missed"))
        }
        labeled = [(self.candidate, {"ring_id": "found"})] * 2
        coverage = metrics.compute_ring_metrics(tags, labeled, [])
        self.assertEqual((coverage.total, coverage.detected, coverage.detection_recall), (2, 1, 0.5))
        self.assertEqual((coverage.flagged, coverage.verdict_recall), (0, 0.0))

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
