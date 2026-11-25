import unittest
from unittest.mock import Mock

from src.detection.detection import detect_circular_flow, detect_shared_device, detect_shared_payout


class DetectionTests(unittest.TestCase):
    def test_shared_device_scores_repeated_common_recipient_activity(self):
        session = Mock()
        accounts = ["C", "B", "A"]
        first_round = [{"sender": a, "recipient": "D", "step": 1} for a in accounts]
        second_round = [{**tx, "step": 25} for tx in first_round]
        session.run.return_value = [
            {"device_id": "DEV-1", "accounts": accounts, "transactions": first_round},
            {"device_id": "DEV-2", "accounts": accounts, "transactions": first_round + second_round},
        ]

        once, repeated = detect_shared_device(session)
        self.assertEqual(repeated.members, ["A", "B", "C"])
        self.assertEqual(repeated.evidence["shared_recipients"], 1)
        self.assertEqual(repeated.evidence["repeat_senders_to_same_recipient"], 3)
        self.assertEqual(repeated.evidence["transactions_per_active_hour"], 3.0)
        self.assertGreater(repeated.suspicion_score, once.suspicion_score)

    def test_shared_payout_counts_distinct_senders_in_a_local_window(self):
        def transfers(*pairs):
            return [{"account_id": account, "step": step} for account, step in pairs]

        session = Mock()
        session.run.return_value = [
            {"dest": "tight", "transfers": transfers(("X", 1), ("A", 100), ("A", 101), ("B", 101), ("C", 102))},
            {"dest": "wide", "transfers": transfers(("A", 100), ("B", 103), ("C", 110))},
            {"dest": "one-sender", "transfers": transfers(("A", 100), ("A", 103), ("A", 110))},
        ]

        tight, wide = detect_shared_payout(session, min_size=3, max_size=6, max_window=12)
        self.assertEqual(session.run.call_args.kwargs, {"min_size": 3})
        self.assertEqual(tight.members, ["A", "B", "C"])
        self.assertEqual(tight.evidence["unique_senders"], 3)
        self.assertEqual(tight.evidence["repeat_senders"], 1)
        self.assertEqual(tight.evidence["transaction_count"], 4)
        self.assertEqual(tight.evidence["step_window"], 2)
        self.assertGreater(tight.suspicion_score, wide.suspicion_score)

    def test_circular_flow_deduplicates_overlapping_laps_and_preserves_direction(self):
        session = Mock()
        session.run.return_value = [
            {"cycle_nodes": ["B", "C", "A"], "amounts": [100, 100, 100], "steps": [1, 2, 3]},
            {"cycle_nodes": ["C", "A", "B"], "amounts": [100, 100, 100], "steps": [2, 3, 9]},
            {"cycle_nodes": ["B", "C", "A"], "amounts": [100, 100, 100], "steps": [9, 10, 11]},
            {"cycle_nodes": ["A", "C", "B"], "amounts": [100, 50, 25], "steps": [1, 5, 10]},
        ]

        forward, reverse = detect_circular_flow(session, max_window=12)
        self.assertEqual(session.run.call_args.kwargs, {"max_window": 12})
        self.assertEqual(forward.members, ["A", "B", "C"])
        self.assertEqual(reverse.members, ["A", "C", "B"])
        self.assertEqual(forward.evidence["amount_cv"], 0.0)
        self.assertEqual(forward.evidence["step_window"], 2)
        self.assertEqual(forward.evidence["cycle_occurrences"], 2)
        self.assertLess(reverse.suspicion_score, forward.suspicion_score)

    def test_circular_flow_scores_the_same_gap_equally_across_cycle_lengths(self):
        session = Mock()
        session.run.return_value = [
            {"cycle_nodes": ["A", "B", "C"], "amounts": [100] * 3, "steps": [0, 6, 12]},
            {"cycle_nodes": ["D", "E", "F", "G", "H", "I"], "amounts": [100] * 6,
             "steps": [0, 6, 12, 18, 24, 30]},
            {"cycle_nodes": ["J", "K", "L"], "amounts": [100] * 3, "steps": [0, 4, 8]},
        ]

        short, long, fast = detect_circular_flow(session)

        self.assertEqual((short.evidence["step_window"], long.evidence["step_window"]), (12, 30))
        self.assertEqual((short.evidence["avg_step_gap"], long.evidence["avg_step_gap"]), (6, 6))
        self.assertEqual(short.evidence["decay_consistency"], 1.0)
        self.assertEqual(short.suspicion_score, 0.6909)
        self.assertEqual(short.suspicion_score, long.suspicion_score)
        self.assertEqual(fast.evidence["avg_step_gap"], 4)
        self.assertGreater(fast.suspicion_score, short.suspicion_score)

    def test_circular_flow_scores_chronological_amount_ratios(self):
        session = Mock()
        scores = {}
        for amounts, expected in [
            ([100, 92, 84.64], 1.0),
            ([100, 50, 25], 1.0),
            ([50, 100, 25], 0.1111),
            ([25, 50, 100], 0.0),
            ([100, 90, 45], 0.7143),
            ([100, 0, 25], 0.0),
            ([0, 0, 0], 0.0),
        ]:
            with self.subTest(amounts=amounts):
                session.run.return_value = [{"cycle_nodes": ["C", "A", "B"], "steps": [1, 2, 3], "amounts": amounts}]
                candidate, = detect_circular_flow(session)
                self.assertEqual(candidate.evidence["decay_consistency"], expected)
                scores[tuple(amounts)] = candidate.suspicion_score
        self.assertAlmostEqual(scores[(100, 50, 25)] - scores[(25, 50, 100)], 0.3)

    def test_circular_flow_records_balance_mismatches_when_all_legs_have_balances(self):
        session = Mock()
        session.run.return_value = [
            {"cycle_nodes": ["A", "B", "C"], "steps": [1, 2, 3], "amounts": [0.02, 100, 100],
             "sender_balances": [[1_000_000, 1_000_000], [100, 0], [100, 4]]},
            {"cycle_nodes": ["D", "E", "F"], "steps": [1, 2, 3], "amounts": [0.02, 100, 100],
             "sender_balances": [[1_000_000, 1_000_000], [None, None], [100, 4]]},
        ]

        complete, partial = detect_circular_flow(session)

        self.assertEqual(complete.sender_balance_mismatch_count, 1)
        self.assertIsNone(partial.sender_balance_mismatch_count)
        self.assertEqual(complete.evidence["sender_balance_mismatch_count"], 1)
        self.assertIsNone(partial.evidence["sender_balance_mismatch_count"])
        self.assertEqual(complete.suspicion_score, partial.suspicion_score)
