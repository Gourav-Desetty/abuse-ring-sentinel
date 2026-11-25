import unittest

import pandas as pd

from src.data.build_graph import _edge_rows
from src.data.plant_rings import PlantConfig, plant_rings


class BalanceTests(unittest.TestCase):
    def test_neo4j_rows_keep_source_balances_and_populate_all_planted_balances(self):
        count = 140
        df = pd.DataFrame({
            "nameOrig": [f"C{i}" for i in range(count)],
            "nameDest": "M0",
            "amount": 12.25,
            "step": 1,
            "type": "PAYMENT",
            "orig_kind": "C",
            "dest_kind": "M",
            "oldbalanceOrg": 31.50,
            "newbalanceOrig": 14.50,
        })
        config = PlantConfig(rings_per_type_per_difficulty=2, negatives_per_method=2)
        graph, rings = plant_rings(df, config)
        rows = _edge_rows(graph, {ring.ring_id: ring for ring in rings})

        original = [row for row in rows if row["ring_id"] is None]
        self.assertEqual(len(original), count)
        for row in original:
            self.assertEqual((row["oldbalanceOrg"], row["newbalanceOrig"]), (31.5, 14.5))

        planted = [row for row in rows if row["ring_id"] is not None]
        self.assertEqual({row["ring_id"] for row in planted}, {ring.ring_id for ring in rings})
        for row in planted:
            self.assertGreaterEqual(row["newbalanceOrig"], 0)
            self.assertAlmostEqual(row["oldbalanceOrg"] - row["newbalanceOrig"], row["amount"])
