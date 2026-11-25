"""Plant account-disjoint collusion rings and hard negatives in PaySim."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import repeat

import networkx as nx
import pandas as pd

DIFFICULTIES = ("obvious", "subtle")
METHODS = ("shared_device", "shared_payout", "circular_flow")
BENIGN_METHODS = {
    "legit_shared_device": "shared_device",
    "legit_personal_transfers": "shared_payout",
    "legit_circular_flow": "circular_flow",
}


@dataclass
class Ring:
    ring_id: str
    ring_type: str
    difficulty: str | None
    members: list[str]
    split: str = "train"


@dataclass
class PlantConfig:
    rings_per_type_per_difficulty: int = 15
    negatives_per_method: int = 15
    min_ring_size: int = 3
    max_ring_size: int = 6
    test_fraction: float = 1 / 3
    random_seed: int = 42


def build_base_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """Build a directed multigraph of nameOrig -> nameDest transactions."""
    g = nx.MultiDiGraph()
    columns = zip(
        df["nameOrig"], df["nameDest"], df["amount"], df["step"], df["type"],
        df["orig_kind"], df["dest_kind"],
        df.get("oldbalanceOrg", repeat(None)), df.get("newbalanceOrig", repeat(None)),
    )
    for name_orig, name_dest, amount, step, tx_type, orig_kind, dest_kind, old_balance, new_balance in columns:
        g.add_node(name_orig, kind=orig_kind)
        g.add_node(name_dest, kind=dest_kind)
        g.add_edge(
            name_orig, name_dest, amount=float(amount), step=int(step), type=tx_type, planted=False,
            oldbalanceOrg=float(old_balance) if pd.notna(old_balance) else None,
            newbalanceOrig=float(new_balance) if pd.notna(new_balance) else None,
        )
    return g


def _plant_pattern(
    g: nx.MultiDiGraph, df: pd.DataFrame, rng: random.Random, balance_rng: random.Random,
    ring: Ring, method: str,
) -> None:
    members = ring.members
    benign = ring.difficulty is None
    subtle = ring.difficulty == "subtle"
    base = rng.randint(1, 600)
    transfers: list[tuple[int, str, str, int]] = []

    def transfer(sender: str, recipient: str, amount: float, step: int) -> None:
        transfers.append((step, sender, recipient, round(round(amount, 2) * 100)))

    amount = float(df["amount"].iloc[rng.randrange(len(df))])
    if method == "shared_device":
        for member in members:
            g.nodes[member]["device_id"] = f"DEV-{members[0]}"
        # A household can pay a common recipient in the same period, including on repeat occasions.
        rounds = rng.choice((1, 1, 2)) if benign else (2 if subtle else 3)
        for visit in range(rounds):
            for member in members:
                transfer(
                    member, f"C-CONTEXT-{members[0]}", amount * rng.uniform(0.2, 1.8),
                    base + visit * 24 + rng.randint(0, 18 if benign or subtle else 2),
                )
    elif method == "shared_payout":
        senders, recipient = members[:-1], members[-1]
        # Group expenses can produce a tight burst. Some participants also reimburse in installments.
        rounds = rng.choice((1, 1, 2)) if benign else (2 if subtle else 3)
        spread = rng.choice((3, 12, 28)) if benign else (25 if subtle else 3)
        for i in range(rounds):
            for sender in senders:
                step = base + i * 12 + rng.randint(0, spread)
                transfer(sender, recipient, amount * rng.uniform(0.4, 1.6) / rounds, step)
    else:
        # Chains of reimbursements can close a loop, sometimes quickly and with similar amounts.
        gap = rng.choice((1, 4, 12)) if benign else (rng.randint(6, 19) if subtle else 1)
        rounds = 1 if benign or subtle else 3
        for lap in range(rounds):
            for i, member in enumerate(members):
                factor = rng.uniform(0.55, 1.3) if benign else (0.92 ** i if subtle else 1)
                transfer(member, members[(i + 1) % len(members)], amount * factor, base + lap * 24 + i * gap)

    # Give senders enough funds for the scheduled transfers plus a sampled reserve.
    # The synthetic ledger covers only these transfers and uses cents to avoid drift.
    balances: defaultdict[str, int] = defaultdict(int)
    for _, sender, _, cents in transfers:
        balances[sender] += cents
    observed_reserves = df.get("newbalanceOrig")
    if observed_reserves is not None:
        for sender in balances:
            reserve = observed_reserves.iloc[balance_rng.randrange(len(df))]
            if pd.notna(reserve):
                balances[sender] += round(max(float(reserve), 0) * 100)

    for step, sender, recipient, cents in sorted(transfers, key=lambda tx: tx[0]):
        old_balance = balances[sender]
        balances[sender] -= cents
        balances[recipient] += cents
        g.add_edge(
            sender, recipient, amount=cents / 100, step=step,
            type="TRANSFER", planted=True, ring_id=ring.ring_id,
            oldbalanceOrg=old_balance / 100, newbalanceOrig=balances[sender] / 100,
        )


def plant_rings(
    df: pd.DataFrame, config: PlantConfig | None = None,
) -> tuple[nx.MultiDiGraph, list[Ring]]:
    """Return a seeded graph with disjoint, stratified calibration/test cohorts."""
    config = config or PlantConfig()
    rng = random.Random(config.random_seed)
    balance_rng = random.Random(f"{config.random_seed}:balances")
    g = build_base_graph(df)
    accounts = [n for n, data in g.nodes(data=True) if data["kind"] == "C"]
    rng.shuffle(accounts)
    rings: list[Ring] = []
    strata = [
        (method, method, difficulty, config.rings_per_type_per_difficulty)
        for method in METHODS
        for difficulty in DIFFICULTIES
    ] + [(kind, method, None, config.negatives_per_method) for kind, method in BENIGN_METHODS.items()]

    for kind, method, difficulty, count in strata:
        for _ in range(count):
            size = rng.randint(config.min_ring_size, config.max_ring_size)
            # Give both payout classes the same recipient role, drawn from the same account pool.
            n_members = size + (method == "shared_payout")
            if len(accounts) < n_members:
                raise ValueError("PaySim sample has too few distinct customer accounts for the benchmark.")
            members = [accounts.pop() for _ in range(n_members)]
            if method == "shared_device":
                g.add_node(f"C-CONTEXT-{members[0]}", kind="C")
            ring_id = f"{kind}-{difficulty or 'benign'}-{len(rings) + 1:03d}"
            ring = Ring(ring_id, kind, difficulty, members)
            _plant_pattern(g, df, rng, balance_rng, ring, method)
            rings.append(ring)

    _assign_splits(rings, config.test_fraction, rng)
    return g, rings


def _assign_splits(rings: list[Ring], test_fraction: float, rng: random.Random) -> None:
    """Stratify by ring type and difficulty with at least one group in each split."""
    groups: dict[tuple[str, str | None], list[Ring]] = defaultdict(list)
    for ring in rings:
        groups[(ring.ring_type, ring.difficulty)].append(ring)

    for group in groups.values():
        rng.shuffle(group)
        n_test = min(len(group) - 1, max(1, round(len(group) * test_fraction)))
        for ring in group[:n_test]:
            ring.split = "test"


def rings_to_frame(rings: list[Ring]) -> pd.DataFrame:
    """Flatten planted rings into one row per (ring, member)."""
    return pd.DataFrame.from_records([
        {
            "ring_id": ring.ring_id,
            "ring_type": ring.ring_type,
            "difficulty": ring.difficulty,
            "split": ring.split,
            "size": len(ring.members),
            "node": member,
        }
        for ring in rings
        for member in ring.members
    ])


def summarize(g: nx.MultiDiGraph, rings: list[Ring]) -> None:
    print(f"graph: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")
    planted_edges = sum(1 for _, _, d in g.edges(data=True) if d.get("planted"))
    print(f"planted edges: {planted_edges:,}")
    frame = rings_to_frame(rings)
    print(f"\ncohorts planted: {len(rings)}")
    print(frame.groupby(["ring_type", "difficulty", "split"], dropna=False).agg(
        n_rings=("ring_id", "nunique"), n_members=("node", "count")
    ))


if __name__ == "__main__":
    from src.data.load_paysim import load_paysim

    frame = load_paysim(nrows=200_000)
    graph, planted_rings = plant_rings(frame)
    summarize(graph, planted_rings)
