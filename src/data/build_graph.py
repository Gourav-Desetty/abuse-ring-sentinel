"""Push the base transaction graph + planted collusion rings into Neo4j.

Consumes load_paysim.py + plant_rings.py's output (the built graph and the
list of planted Rings) and writes it via UNWIND-batched, idempotent
(MERGE-on-account_id / MERGE-on-defining-properties) writes -- see
ARCHITECTURE.md § Phase 1 for the full node/relationship schema and env var
setup.
"""

from __future__ import annotations

import argparse
import itertools
import os
from dataclasses import dataclass
from typing import Iterator, LiteralString

import networkx as nx
from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase, Session

from src.data.load_paysim import load_paysim
from src.data.plant_rings import Ring, plant_rings

DEFAULT_BATCH_SIZE = 5_000


@dataclass
class Neo4jConfig:
    uri: str
    username: str
    password: str
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        load_dotenv()
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        if not uri or not username or not password:
            raise RuntimeError(
                "NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must be set "
                "(environment or .env) to load the graph."
            )
        return cls(uri=uri, username=username, password=password, database=os.getenv("NEO4J_DATABASE", "neo4j"))


def get_driver(config: Neo4jConfig) -> Driver:
    return GraphDatabase.driver(config.uri, auth=(config.username, config.password))


def _chunks(rows: list[dict], size: int) -> Iterator[list[dict]]:
    it = iter(rows)
    while chunk := list(itertools.islice(it, size)):
        yield chunk


def create_constraints(driver: Driver, config: Neo4jConfig) -> None:
    """Create the Account.account_id uniqueness constraint (and its backing
    index) up front -- required for MERGE to stay fast at 6M+ nodes, and
    for node idempotency across re-runs.
    """
    with driver.session(database=config.database) as session:
        session.run(
            "CREATE CONSTRAINT account_id_unique IF NOT EXISTS "
            "FOR (a:Account) REQUIRE a.account_id IS UNIQUE"
        )


def _ring_tag_map(rings: list[Ring]) -> dict[str, dict]:
    """node_id -> {ring_id, ring_type, difficulty, split} for every planted
    ring member. A node planted into more than one ring keeps whichever
    ring's tag is written last -- a rare collision, acceptable here.
    """
    tags: dict[str, dict] = {}
    for ring in rings:
        for member in ring.members:
            tags[member] = {
                "ring_id": ring.ring_id,
                "ring_type": ring.ring_type,
                "difficulty": ring.difficulty,
                "split": ring.split,
            }
    return tags


def _node_rows(g: nx.MultiDiGraph, ring_tags: dict[str, dict]) -> list[dict]:
    rows = []
    for node, data in g.nodes(data=True):
        tag = ring_tags.get(node, {})
        rows.append(
            {
                "account_id": node,
                "kind": data.get("kind"),
                "device_id": data.get("device_id"),
                "ring_id": tag.get("ring_id"),
                "ring_type": tag.get("ring_type"),
                "difficulty": tag.get("difficulty"),
                "split": tag.get("split"),
            }
        )
    return rows


def _edge_rows(g: nx.MultiDiGraph, ring_lookup: dict[str, Ring]) -> list[dict]:
    rows = []
    for u, v, data in g.edges(data=True):
        ring_id = data.get("ring_id")
        ring = ring_lookup.get(ring_id) if ring_id else None
        rows.append(
            {
                "from_id": u,
                "to_id": v,
                "amount": data.get("amount"),
                "step": data.get("step"),
                "type": data.get("type"),
                "ring_id": ring_id,
                "ring_type": ring.ring_type if ring else None,
                "difficulty": ring.difficulty if ring else None,
                "split": ring.split if ring else None,
            }
        )
    return rows


def _shared_payout_rows(g: nx.MultiDiGraph, rings: list[Ring]) -> list[dict]:
    """Derive SHARES_PAYOUT pairs for each shared_payout ring from its
    planted TRANSACTED_WITH legs, rather than from `Ring.members` directly,
    so the payout target is identified structurally (whoever fed it) instead
    of by naming convention.
    """
    rows = []
    for ring in rings:
        if ring.ring_type != "shared_payout":
            continue
        payers = sorted({u for u, _, data in g.edges(data=True) if data.get("ring_id") == ring.ring_id})
        for a, b in itertools.combinations(payers, 2):
            rows.append(
                {
                    "a_id": a,
                    "b_id": b,
                    "ring_id": ring.ring_id,
                    "ring_type": ring.ring_type,
                    "difficulty": ring.difficulty,
                    "split": ring.split,
                }
            )
    return rows


_NODE_QUERY: LiteralString = """
UNWIND $rows AS row
MERGE (a:Account {account_id: row.account_id})
SET a.kind = row.kind,
    a.device_id = row.device_id,
    a.ring_id = row.ring_id,
    a.ring_type = row.ring_type,
    a.difficulty = row.difficulty,
    a.split = row.split
"""

_TRANSACTED_WITH_QUERY: LiteralString = """
UNWIND $rows AS row
MATCH (a:Account {account_id: row.from_id})
MATCH (b:Account {account_id: row.to_id})
MERGE (a)-[r:TRANSACTED_WITH {amount: row.amount, step: row.step, type: row.type}]->(b)
SET r.ring_id = row.ring_id,
    r.ring_type = row.ring_type,
    r.difficulty = row.difficulty,
    r.split = row.split
"""

_SHARES_PAYOUT_QUERY: LiteralString = """
UNWIND $rows AS row
MATCH (a:Account {account_id: row.a_id})
MATCH (b:Account {account_id: row.b_id})
MERGE (a)-[r:SHARES_PAYOUT {ring_id: row.ring_id}]->(b)
SET r.ring_type = row.ring_type,
    r.difficulty = row.difficulty,
    r.split = row.split
"""


def _run_batched(driver: Driver, config: Neo4jConfig, query: LiteralString, rows: list[dict], batch_size: int) -> None:
    with driver.session(database=config.database) as session:
        for chunk in _chunks(rows, batch_size):
            session.run(query, rows=chunk)


def build_graph(
    driver: Driver,
    config: Neo4jConfig,
    g: nx.MultiDiGraph,
    rings: list[Ring],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    create_constraints(driver, config)

    ring_tags = _ring_tag_map(rings)
    ring_lookup = {ring.ring_id: ring for ring in rings}

    node_rows = _node_rows(g, ring_tags)
    edge_rows = _edge_rows(g, ring_lookup)
    payout_rows = _shared_payout_rows(g, rings)

    print(f"loading {len(node_rows):,} Account nodes...")
    _run_batched(driver, config, _NODE_QUERY, node_rows, batch_size)

    print(f"loading {len(edge_rows):,} TRANSACTED_WITH relationships...")
    _run_batched(driver, config, _TRANSACTED_WITH_QUERY, edge_rows, batch_size)

    print(f"loading {len(payout_rows):,} SHARES_PAYOUT relationships...")
    _run_batched(driver, config, _SHARES_PAYOUT_QUERY, payout_rows, batch_size)


def _scalar(session: Session, query: LiteralString) -> int:
    """Run a `RETURN ... AS n` query and return that single count.

    `.single(strict=True)` raises rather than returning None if the query
    didn't yield exactly one record, so a plain `[...]` index here is safe
    -- unlike `.single()`, which types as `Record | None`.
    """
    record = session.run(query).single(strict=True)
    assert record is not None  # strict=True raises rather than returning None
    return record["n"]


def verify_load(driver: Driver, config: Neo4jConfig, g: nx.MultiDiGraph, rings: list[Ring]) -> None:
    """Print sanity counts from Neo4j and cross-check them against the
    in-memory graph/rings that were just loaded -- the same numbers
    plant_rings.py's own summarize() prints, so a mismatch here means the
    load (not the planting) introduced a bug.
    """
    expected_nodes = g.number_of_nodes()
    expected_edges = g.number_of_edges()
    expected_device_tagged = sum(1 for _, d in g.nodes(data=True) if "device_id" in d)
    expected_planted_edges = sum(1 for _, _, d in g.edges(data=True) if d.get("planted"))

    with driver.session(database=config.database) as session:
        n_accounts = _scalar(session, "MATCH (a:Account) RETURN count(a) AS n")
        n_tx = _scalar(session, "MATCH ()-[r:TRANSACTED_WITH]->() RETURN count(r) AS n")
        n_payout = _scalar(session, "MATCH ()-[r:SHARES_PAYOUT]->() RETURN count(r) AS n")
        n_device = _scalar(session, "MATCH (a:Account) WHERE a.device_id IS NOT NULL RETURN count(a) AS n")
        n_ring_nodes = _scalar(session, "MATCH (a:Account) WHERE a.ring_id IS NOT NULL RETURN count(a) AS n")
        n_ring_tx = _scalar(
            session, "MATCH ()-[r:TRANSACTED_WITH]->() WHERE r.ring_id IS NOT NULL RETURN count(r) AS n"
        )
        breakdown = list(
            session.run(
                "MATCH (a:Account) WHERE a.ring_id IS NOT NULL "
                "RETURN a.ring_type AS ring_type, a.difficulty AS difficulty, a.split AS split, "
                "count(a) AS n_nodes "
                "ORDER BY ring_type, difficulty, split"
            )
        )

    print(f"\nNeo4j Account nodes: {n_accounts:,}  (graph had {expected_nodes:,})")
    print(f"Neo4j TRANSACTED_WITH rels: {n_tx:,}  (graph had {expected_edges:,})")
    print(f"Neo4j SHARES_PAYOUT rels: {n_payout:,}")
    print(f"Neo4j device-tagged accounts: {n_device:,}  (expected {expected_device_tagged:,})")
    print(f"Neo4j ring-tagged accounts: {n_ring_nodes:,}")
    print(f"Neo4j ring-tagged TRANSACTED_WITH rels: {n_ring_tx:,}  (expected {expected_planted_edges:,})")
    print(f"planted rings loaded: {len(rings)}")

    print("\nring-tagged accounts by ring_type / difficulty / split:")
    for row in breakdown:
        print(f"  {row['ring_type']:<15} {str(row['difficulty']):<8} {row['split']:<6} n_nodes={row['n_nodes']}")

    ok = (
        n_accounts == expected_nodes
        and n_tx == expected_edges
        and n_device == expected_device_tagged
        and n_ring_tx == expected_planted_edges
    )
    print(f"\nsanity check: {'PASS' if ok else 'MISMATCH -- see counts above'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nrows", type=int, default=None, help="load only the first N PaySim rows (fast dev iteration)")
    parser.add_argument(
        "--sample-frac", type=float, default=None, help="randomly sample this fraction of loaded rows before building the graph"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"UNWIND batch size (default {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--wipe", action="store_true", help="delete all nodes/relationships before loading (clean test run)")
    args = parser.parse_args(argv)

    config = Neo4jConfig.from_env()
    driver = get_driver(config)

    try:
        if args.wipe:
            print("wiping existing graph...")
            with driver.session(database=config.database) as session:
                session.run("MATCH (n) DETACH DELETE n")

        df = load_paysim(nrows=args.nrows, sample_frac=args.sample_frac)
        g, rings = plant_rings(df)

        build_graph(driver, config, g, rings, batch_size=args.batch_size)
        verify_load(driver, config, g, rings)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
