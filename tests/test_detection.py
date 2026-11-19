import unittest
from unittest.mock import Mock

from src.detection.detection import detect_circular_flow, detect_shared_device, detect_shared_payout


class DetectionTests(unittest.TestCase):
    def test_shared_device_groups_accounts_and_caps_score(self):
        session = Mock()
        session.run.return_value = [{"device_id": "DEV-1", "accounts": ["E", "D", "C", "B", "A"]}]

        candidate, = detect_shared_device(session)

        self.assertEqual(candidate.candidate_id, "shared_device::DEV-1")
        self.assertEqual(candidate.members, ["A", "B", "C", "D", "E"])
        self.assertEqual(candidate.evidence, {"device_id": "DEV-1", "size": 5})
        self.assertEqual(candidate.suspicion_score, 1.0)

    def test_shared_payout_passes_query_bounds_and_rewards_tighter_fan_in(self):
        payers = [{"account_id": "C3"}, {"account_id": "C1"}, {"account_id": "C2"}]
        session = Mock()
        session.run.return_value = [
            {"dest": "C-tight", "payers": payers, "step_window": 0},
            {"dest": "C-wide", "payers": payers, "step_window": 10},
        ]

        tight, wide = detect_shared_payout(session, min_size=3, max_size=6, max_window=12)

        self.assertEqual(session.run.call_args.kwargs, {"min_size": 3, "max_size": 6, "max_window": 12})
        self.assertEqual(tight.members, ["C1", "C2", "C3"])
        self.assertEqual(wide.evidence, {"payout_account": "C-wide", "size": 3, "step_window": 10})
        self.assertEqual((tight.suspicion_score, wide.suspicion_score), (0.5, 0.25))

    def test_circular_flow_deduplicates_rotations_and_preserves_direction(self):
        session = Mock()
        session.run.return_value = [
            {"cycle_nodes": ["B", "C", "A", "B"], "amounts": [100, 100, 100]},
            {"cycle_nodes": ["C", "A", "B", "C"], "amounts": [100, 100, 100]},
            {"cycle_nodes": ["A", "C", "B", "A"], "amounts": [100, 50, 25]},
        ]

        forward, reverse = detect_circular_flow(session)

        self.assertEqual(forward.members, ["A", "B", "C"])
        self.assertEqual(reverse.members, ["A", "C", "B"])
        self.assertEqual(forward.evidence["amount_cv"], 0.0)
        self.assertEqual(forward.suspicion_score, 1.0)
        self.assertLess(reverse.suspicion_score, forward.suspicion_score)
