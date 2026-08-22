"""Plant synthetic collusion rings on top of a PaySim-derived transaction graph.

PaySim's ``isFraud``/``isFlaggedFraud`` are not ring labels (see
``load_paysim.py``), so ring ground truth has to be manufactured: this module
builds a directed transaction graph from real PaySim edges and stamps three
kinds of collusion ring onto it, each at "obvious" and "subtle" difficulty so
precision/recall on the result is meaningful rather than trivially perfect:

- **shared_device**: multiple accounts tagged with the same synthetic device
  ID.
- **shared_payout**: multiple accounts routing funds into the same
  downstream payout account.
- **circular_flow**: an A -> B -> C -> ... -> A transaction cycle.

It also plants **legit_cluster** groups that superficially resemble
shared_payout rings (many customers paying one popular merchant) but are not
collusive, to act as hard negatives -- without them, precision/recall would
be measured against an unrealistically easy problem.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx
import pandas as pd

DIFFICULTIES = ("obvious", "subtle")


@dataclass
class Ring:
    ring_id: str
    ring_type: str  # "shared_device" | "shared_payout" | "circular_flow" | "legit_cluster"
    difficulty: str | None  # "obvious" | "subtle", None for legit_cluster
    members: list[str]
    split: str = "train"  # "train" | "test", assigned by hold-out step


@dataclass
class PlantConfig:
    rings_per_type_per_difficulty: int = 4
    legit_clusters: int = 6
    min_ring_size: int = 3
    max_ring_size: int = 6
    test_fraction: float = 0.3
    random_seed: int = 42


def build_base_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """Build a directed multigraph of nameOrig -> nameDest transactions."""
    g = nx.MultiDiGraph()
    columns = zip(
        df["nameOrig"], df["nameDest"], df["amount"], df["step"], df["type"],
        df["orig_kind"], df["dest_kind"],
    )
    for name_orig, name_dest, amount, step, tx_type, orig_kind, dest_kind in columns:
        g.add_node(name_orig, kind=orig_kind)
        g.add_node(name_dest, kind=dest_kind)
        g.add_edge(
            name_orig,
            name_dest,
            amount=float(amount),
            step=int(step),
            type=tx_type,
            planted=False,
        )
    return g


def _amount_sample(df: pd.DataFrame, rng: random.Random) -> float:
    """Draw a realistic amount from the real transaction-amount distribution."""
    idx = rng.randrange(len(df))
    return float(df["amount"].iloc[idx])


def _customer_nodes(g: nx.MultiDiGraph) -> list[str]:
    return [n for n, d in g.nodes(data=True) if d.get("kind") == "C"]


def _merchant_nodes(g: nx.MultiDiGraph) -> list[str]:
    return [n for n, d in g.nodes(data=True) if d.get("kind") == "M"]


def plant_shared_device_ring(
    g: nx.MultiDiGraph,
    df: pd.DataFrame,
    rng: random.Random,
    ring_id: str,
    size: int,
    difficulty: str,
) -> Ring:
    """Tag `size` existing customer accounts with the same synthetic device ID.

    Obvious: members also make a payment to a common merchant within a
    tight step window, so both the device link and a transaction pattern
    point at the ring. Subtle: only the device_id node attribute is shared
    -- no accompanying transaction pattern.
    """
    candidates = _customer_nodes(g)
    members = rng.sample(candidates, k=size)
    device_id = f"DEV-{ring_id}"
    for node in members:
        g.nodes[node]["device_id"] = device_id

    if difficulty == "obvious":
        shared_merchant = rng.choice(_merchant_nodes(g))
        base_step = rng.randint(1, 700)
        for node in members:
            g.add_edge(
                node,
                shared_merchant,
                amount=_amount_sample(df, rng),
                step=base_step + rng.randint(0, 3),
                type="PAYMENT",
                planted=True,
                ring_id=ring_id,
            )

    return Ring(ring_id, "shared_device", difficulty, members)


def plant_shared_payout_ring(
    g: nx.MultiDiGraph,
    df: pd.DataFrame,
    rng: random.Random,
    ring_id: str,
    size: int,
    difficulty: str,
) -> Ring:
    """Route `size` customer accounts into the same new downstream payout account.

    Obvious: all legs land in a tight step window (fast fan-in). Subtle:
    legs are spread over a wide window, closer to coincidental timing.
    """
    members = rng.sample(_customer_nodes(g), k=size)
    payout_node = f"C-PAYOUT-{ring_id}"
    g.add_node(payout_node, kind="C")

    base_step = rng.randint(1, 700)
    window = 5 if difficulty == "obvious" else 60
    for node in members:
        g.add_edge(
            node,
            payout_node,
            amount=_amount_sample(df, rng),
            step=base_step + rng.randint(0, window),
            type="TRANSFER",
            planted=True,
            ring_id=ring_id,
        )

    return Ring(ring_id, "shared_payout", difficulty, members + [payout_node])


def plant_circular_flow_ring(
    g: nx.MultiDiGraph,
    df: pd.DataFrame,
    rng: random.Random,
    ring_id: str,
    size: int,
    difficulty: str,
) -> Ring:
    """Create an A -> B -> C -> ... -> A transaction cycle.

    Obvious: every leg carries (near) the same amount in quick succession --
    classic layering. Subtle: amount shrinks a little each hop (as if a cut
    is taken) and legs are spread further apart in time.
    """
    members = rng.sample(_customer_nodes(g), k=size)

    base_step = rng.randint(1, 700)
    step_gap = 1 if difficulty == "obvious" else rng.randint(5, 20)
    amount = _amount_sample(df, rng)
    for i, node in enumerate(members):
        nxt = members[(i + 1) % len(members)]
        leg_amount = amount if difficulty == "obvious" else amount * rng.uniform(0.85, 0.98) ** i
        g.add_edge(
            node,
            nxt,
            amount=leg_amount,
            step=base_step + i * step_gap,
            type="TRANSFER",
            planted=True,
            ring_id=ring_id,
        )

    return Ring(ring_id, "circular_flow", difficulty, members)


def plant_legit_cluster(
    g: nx.MultiDiGraph,
    df: pd.DataFrame,
    rng: random.Random,
    ring_id: str,
    size: int,
) -> Ring:
    """Plant a non-collusive hard negative: many customers paying one popular
    merchant, spread widely over time with no shared device -- superficially
    similar fan-in to shared_payout, but nothing colludes them.
    """
    members = rng.sample(_customer_nodes(g), k=size)
    merchant = rng.choice(_merchant_nodes(g))
    for node in members:
        g.add_edge(
            node,
            merchant,
            amount=_amount_sample(df, rng),
            step=rng.randint(1, 743),
            type="PAYMENT",
            planted=True,
            ring_id=ring_id,
        )
    return Ring(ring_id, "legit_cluster", None, members + [merchant])


_PLANTERS = {
    "shared_device": plant_shared_device_ring,
    "shared_payout": plant_shared_payout_ring,
    "circular_flow": plant_circular_flow_ring,
}


def plant_rings(
    df: pd.DataFrame,
    config: PlantConfig | None = None,
) -> tuple[nx.MultiDiGraph, list[Ring]]:
    """Build the base graph from `df` and plant collusion rings + legit clusters.

    Returns the mutated graph and the list of planted `Ring`s (with `.split`
    set to "train"/"test" per `config.test_fraction`, stratified by
    ring_type + difficulty so both splits contain every kind of ring).
    """
    config = config or PlantConfig()
    rng = random.Random(config.random_seed)

    g = build_base_graph(df)
    rings: list[Ring] = []

    counter = 0
    for ring_type, planter in _PLANTERS.items():
        for difficulty in DIFFICULTIES:
            for _ in range(config.rings_per_type_per_difficulty):
                counter += 1
                ring_id = f"{ring_type}-{difficulty}-{counter:03d}"
                size = rng.randint(config.min_ring_size, config.max_ring_size)
                rings.append(planter(g, df, rng, ring_id, size, difficulty))

    for _ in range(config.legit_clusters):
        counter += 1
        ring_id = f"legit_cluster-{counter:03d}"
        size = rng.randint(config.min_ring_size, config.max_ring_size)
        rings.append(plant_legit_cluster(g, df, rng, ring_id, size))

    _assign_splits(rings, config.test_fraction, rng)

    return g, rings


def _assign_splits(rings: list[Ring], test_fraction: float, rng: random.Random) -> None:
    """Stratify train/test by (ring_type, difficulty) so both splits contain
    every kind of ring, and never used for threshold tuning once assigned.
    """
    from collections import defaultdict

    groups: dict[tuple[str, str | None], list[Ring]] = defaultdict(list)
    for ring in rings:
        groups[(ring.ring_type, ring.difficulty)].append(ring)

    for group in groups.values():
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_fraction)) if len(group) > 1 else 0
        for ring in group[:n_test]:
            ring.split = "test"
        for ring in group[n_test:]:
            ring.split = "train"


def rings_to_frame(rings: list[Ring]) -> pd.DataFrame:
    """Flatten planted rings into one row per (ring, member) for inspection/eval."""
    records = [
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
    ]
    return pd.DataFrame.from_records(records)


def summarize(g: nx.MultiDiGraph, rings: list[Ring]) -> None:
    """Print a quick sanity report -- used to eyeball plant_rings's output."""
    print(f"graph: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")
    planted_edges = sum(1 for _, _, d in g.edges(data=True) if d.get("planted"))
    print(f"planted edges: {planted_edges:,}")

    frame = rings_to_frame(rings)
    print(f"\nrings planted: {len(rings)}")
    print(frame.groupby(["ring_type", "difficulty", "split"], dropna=False).agg(
        n_rings=("ring_id", "nunique"), n_members=("node", "count")
    ))

    device_tagged = sum(1 for _, d in g.nodes(data=True) if "device_id" in d)
    print(f"\naccounts tagged with a planted device_id: {device_tagged}")


if __name__ == "__main__":
    from src.data.load_paysim import load_paysim

    frame = load_paysim(nrows=200_000)
    graph, planted_rings = plant_rings(frame)
    summarize(graph, planted_rings)
