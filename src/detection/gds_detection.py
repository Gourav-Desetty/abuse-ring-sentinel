"""Extract candidate collusion rings from the Neo4j transaction graph.

Phase 2 of PLAN.MD's architecture: "Neo4j GDS (community detection, cycle
detection) -> input guardrail (only high-suspicion candidates pass) -> ...".
This module is the first half -- it surfaces *candidates* with raw graph
evidence and a suspicion score; a later input guardrail (`agent/guardrails.py`,
build order step 4) decides which candidates are suspicious enough to reach
the LLM.

Why plain Cypher instead of the `gds.*` procedure library
-----------------------------------------------------------
PLAN.MD names "Neo4j GDS" for this step, and that's what this module would
normally call (`gds.louvain.stream`, `gds.scc.stream`, ...). But this
project's Neo4j is AuraDB, which ships **Aura Graph Analytics Serverless**
rather than classic self-managed GDS: every `gds.*` algorithm call requires
first provisioning a separate, billed compute session
(`gds.session.getOrCreate`, ~$0.40/GB-hour, ~2GB recommended for this
graph). Rather than spend real money per detection run, the three ring
shapes are detected with equivalent plain-Cypher graph queries instead:

- **shared_device** -> group accounts by the `device_id` node property.
- **shared_payout** -> group accounts whose TRANSACTED_WITH edges fan into
  the same destination account (this is what community detection would
  otherwise be finding: a small, locally dense cluster around one sink).
- **circular_flow** -> a bounded variable-length path query
  `(a)-[:TRANSACTED_WITH*3..6]->(a)` (what gds.scc.stream would otherwise
  give you, minus the parts of a strongly-connected component that aren't
  part of any actual cycle).

At this graph's size (~150K nodes) plain Cypher runs these in low single
digit seconds -- GDS would only start winning at a scale this project isn't
operating at.

IMPORTANT: detection here must not read `ring_id` / `ring_type` /
`difficulty` / `split` or the `:SHARES_PAYOUT` relationship -- those are
ground truth stamped by `plant_rings.py` / `build_graph.py` for evaluation
only. A real detector never sees them; using them here would make detection
trivial and `eval/metrics.py`'s future precision/recall numbers meaningless.
Detection works only off `:TRANSACTED_WITH` structure and the `device_id`
property (standing in for a real device-fingerprint signal, not a ring
label). `evaluate_against_ground_truth()` below is the one exception: it
cross-checks detector recall against ring_id for developer sanity-checking,
the same way `build_graph.py`'s `verify_load()` cross-checks load counts --
it is not part of the detector and is not the project's formal evaluation
(that's `eval/metrics.py`, a later build-order step).
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import LiteralString, cast

from neo4j import Driver, Session

from src.data.build_graph import Neo4jConfig, get_driver

MIN_FANIN_SIZE = 3
MAX_FANIN_SIZE = 8  # matches plant_rings.py's ring-size range (3-6), with slack
MAX_FANIN_STEP_WINDOW = 60  # matches plant_rings.py's "subtle" shared_payout window
MIN_CYCLE_LEN = 3
MAX_CYCLE_LEN = 6


@dataclass
class Candidate:
    candidate_id: str
    method: str  # "shared_device" | "shared_payout" | "circular_flow"
    members: list[str]
    evidence: dict
    suspicion_score: float  # roughly 0-1, higher = more suspicious


_SHARED_DEVICE_QUERY: LiteralString = """
MATCH (a:Account)
WHERE a.device_id IS NOT NULL
WITH a.device_id AS device_id, collect(a.account_id) AS accounts
WHERE size(accounts) > 1
RETURN device_id, accounts
"""


def detect_shared_device(session: Session) -> list[Candidate]:
    """Accounts sharing a device_id -- a strong, near-unambiguous signal on
    its own (unrelated accounts colliding on a synthetic device id is
    vanishingly unlikely), so every group gets a high suspicion score.
    """
    candidates = []
    for row in session.run(_SHARED_DEVICE_QUERY):
        accounts = sorted(row["accounts"])
        candidates.append(
            Candidate(
                candidate_id=f"shared_device::{row['device_id']}",
                method="shared_device",
                members=accounts,
                evidence={"device_id": row["device_id"], "size": len(accounts)},
                suspicion_score=min(1.0, 0.6 + 0.1 * len(accounts)),
            )
        )
    return candidates


_SHARED_PAYOUT_QUERY: LiteralString = """
MATCH (a:Account)-[r:TRANSACTED_WITH]->(dest:Account)
WHERE dest.kind = 'C' AND r.type = 'TRANSFER'
WITH dest, collect({account_id: a.account_id, step: r.step}) AS payers
WITH dest, payers, size(payers) AS n,
     reduce(mx = -1, p IN payers | CASE WHEN p.step > mx THEN p.step ELSE mx END) AS max_step,
     reduce(mn = 999999, p IN payers | CASE WHEN p.step < mn THEN p.step ELSE mn END) AS min_step
WITH dest, payers, n, max_step - min_step AS step_window
WHERE n >= $min_size AND n <= $max_size AND step_window <= $max_window
RETURN dest.account_id AS dest, payers, step_window
"""


def detect_shared_payout(
    session: Session,
    min_size: int = MIN_FANIN_SIZE,
    max_size: int = MAX_FANIN_SIZE,
    max_window: int = MAX_FANIN_STEP_WINDOW,
) -> list[Candidate]:
    """Accounts fanning into the same downstream personal (kind='C') account.

    Raw fan-in count alone is a bad discriminator: PaySim reuses nameDest
    ids enough that thousands of unrelated 'C' accounts organically
    accumulate several payers each (see build_graph.py's SHARES_PAYOUT
    docstring). Filters, all required (not just scored), narrow this to a
    tractable candidate set:
      - dest.kind = 'C', not 'M' -- a merchant naturally accumulates
        thousands of unrelated customers (the legit_cluster hard-negative
        shape); many accounts converging on one *personal* account is
        unusual on its own.
      - r.type = 'TRANSFER' -- account-to-account routing, matching how a
        payout ring actually moves money, as opposed to CASH_OUT/CASH_IN/
        PAYMENT.
      - group size and step window bounded to roughly the range
        plant_rings.py plants (3-8 members, <=60-step window) -- a ring is
        a small, time-clustered group, not a personal account's entire
        transaction history accumulated over the dataset's full span.

    Even with these, a *chronologically truncated* sample (`--nrows N`,
    which only spans PaySim's first N rows -- e.g. --nrows 100000 covers
    just steps 1-10 of 743) makes the step-window filter nearly useless:
    every transaction in the sample is already tightly clustered in time,
    real or planted. Load with `--sample-frac` instead of `--nrows` so the
    graph spans PaySim's full step range and "tight window" is actually a
    meaningful signal.
    """
    candidates = []
    for row in session.run(_SHARED_PAYOUT_QUERY, min_size=min_size, max_size=max_size, max_window=max_window):
        payers = row["payers"]
        accounts = sorted(p["account_id"] for p in payers)
        step_window = row["step_window"]
        # size reward (relative to the max plausible ring size), decayed by
        # how spread out the window is
        score = min(1.0, len(accounts) / max_size) * (1.0 / (1.0 + step_window / 10))
        candidates.append(
            Candidate(
                candidate_id=f"shared_payout::{row['dest']}",
                method="shared_payout",
                members=accounts,
                evidence={"payout_account": row["dest"], "size": len(accounts), "step_window": step_window},
                suspicion_score=round(score, 4),
            )
        )
    return candidates


_CIRCULAR_FLOW_QUERY: LiteralString = """
MATCH p = (a:Account)-[:TRANSACTED_WITH*3..6]->(a)
WITH nodes(p) AS ns, relationships(p) AS rels
RETURN [n IN ns | n.account_id] AS cycle_nodes, [r IN rels | r.amount] AS amounts
"""


def _canonical_cycle(members: list[str]) -> tuple[str, ...]:
    """Rotate a cycle's member list to start at its lexicographically
    smallest account_id, so e.g. [A,B,C] and [B,C,A] (the same cycle found
    from two different starting nodes) collapse to one candidate.
    """
    start = min(range(len(members)), key=lambda i: members[i])
    return tuple(members[start:] + members[:start])


def detect_circular_flow(session: Session, min_len: int = MIN_CYCLE_LEN, max_len: int = MAX_CYCLE_LEN) -> list[Candidate]:
    """A -> B -> ... -> A transaction cycles, bounded to `min_len..max_len`
    hops (matches the size range plant_rings.py plants). Suspicion score
    rewards low variance in leg amounts -- constant-amount cycles are the
    layering signature; real coincidental cycles won't have that.
    """
    # Cypher can't parameterize a variable-length relationship's hop bounds,
    # so they're inlined into the query text -- safe here since min_len/
    # max_len are argparse `type=int` values, never interpolated strings.
    query = cast(LiteralString, _CIRCULAR_FLOW_QUERY.replace("*3..6", f"*{min_len}..{max_len}"))
    seen: dict[tuple[str, ...], Candidate] = {}
    for row in session.run(query):
        cycle_nodes = row["cycle_nodes"][:-1]  # drop the repeated closing node
        amounts = row["amounts"]
        key = _canonical_cycle(cycle_nodes)
        if key in seen:
            continue
        mean_amount = statistics.mean(amounts)
        cv = statistics.pstdev(amounts) / mean_amount if mean_amount else 1.0
        score = max(0.0, 1.0 - cv)
        seen[key] = Candidate(
            candidate_id=f"circular_flow::{'-'.join(key)}",
            method="circular_flow",
            members=list(key),
            evidence={"length": len(key), "amounts": amounts, "amount_cv": round(cv, 4)},
            suspicion_score=round(score, 4),
        )
    return list(seen.values())


def detect_candidates(
    driver: Driver,
    config: Neo4jConfig,
    min_fanin_size: int = MIN_FANIN_SIZE,
    max_fanin_size: int = MAX_FANIN_SIZE,
    max_fanin_step_window: int = MAX_FANIN_STEP_WINDOW,
    min_cycle_len: int = MIN_CYCLE_LEN,
    max_cycle_len: int = MAX_CYCLE_LEN,
) -> list[Candidate]:
    """Run all three detectors and return every raw candidate found, each
    tagged with a suspicion_score. Beyond shared_payout's hard filters
    (see detect_shared_payout), no further thresholding happens here --
    that's the input guardrail's job, downstream.
    """
    with driver.session(database=config.database) as session:
        candidates = []
        candidates += detect_shared_device(session)
        candidates += detect_shared_payout(session, min_size=min_fanin_size, max_size=max_fanin_size, max_window=max_fanin_step_window)
        candidates += detect_circular_flow(session, min_len=min_cycle_len, max_len=max_cycle_len)
    return candidates


def evaluate_against_ground_truth(driver: Driver, config: Neo4jConfig, candidates: list[Candidate]) -> None:
    """Developer sanity-check only (see module docstring): for each planted
    ring, report whether any detected candidate of the matching method
    overlaps its members. Not the project's formal precision/recall
    (that's eval/metrics.py) -- just a quick "did detection even work" spot
    check, the same role build_graph.py's verify_load() plays for loading.
    """
    with driver.session(database=config.database) as session:
        rings = list(
            session.run(
                "MATCH (a:Account) WHERE a.ring_id IS NOT NULL "
                "RETURN a.ring_id AS ring_id, a.ring_type AS ring_type, a.difficulty AS difficulty, "
                "a.split AS split, collect(a.account_id) AS members"
            )
        )

    by_method: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_method.setdefault(c.method, []).append(c)

    print("\nrecall spot-check (developer sanity only -- see eval/metrics.py for real metrics):")
    hits, total = 0, 0
    for ring in rings:
        ring_type, members = ring["ring_type"], set(ring["members"])
        if ring_type == "legit_cluster":
            continue  # not a positive to recall -- it's the hard negative
        total += 1
        found = any(set(c.members) & members for c in by_method.get(ring_type, []))
        hits += found
        status = "FOUND" if found else "MISSED"
        print(f"  [{status}] {ring['ring_id']:<28} ({ring['difficulty']}, {ring['split']})")
    print(f"\n{hits}/{total} planted (non-legit_cluster) rings had an overlapping candidate")


def summarize(candidates: list[Candidate]) -> None:
    print(f"candidates found: {len(candidates)}")
    by_method: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_method.setdefault(c.method, []).append(c)
    for method, group in sorted(by_method.items()):
        scores = sorted((c.suspicion_score for c in group), reverse=True)
        print(f"  {method:<15} n={len(group):<6} top scores={scores[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-fanin-size", type=int, default=MIN_FANIN_SIZE)
    parser.add_argument("--max-fanin-size", type=int, default=MAX_FANIN_SIZE)
    parser.add_argument("--max-fanin-step-window", type=int, default=MAX_FANIN_STEP_WINDOW)
    parser.add_argument("--min-cycle-len", type=int, default=MIN_CYCLE_LEN)
    parser.add_argument("--max-cycle-len", type=int, default=MAX_CYCLE_LEN)
    parser.add_argument("--check-ground-truth", action="store_true", help="run the developer recall spot-check after detection")
    args = parser.parse_args()

    config = Neo4jConfig.from_env()
    driver = get_driver(config)
    try:
        candidates = detect_candidates(
            driver,
            config,
            min_fanin_size=args.min_fanin_size,
            max_fanin_size=args.max_fanin_size,
            max_fanin_step_window=args.max_fanin_step_window,
            min_cycle_len=args.min_cycle_len,
            max_cycle_len=args.max_cycle_len,
        )
        summarize(candidates)
        if args.check_ground_truth:
            evaluate_against_ground_truth(driver, config, candidates)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
