"""Find shared devices, payout fan-in, and transaction cycles in Neo4j using Cypher."""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from typing import LiteralString, cast

from neo4j import Driver, Session

from src.data.build_graph import Neo4jConfig, get_driver

MIN_FANIN_SIZE = 3
MAX_FANIN_SIZE = 8
MAX_FANIN_STEP_WINDOW = 60
MIN_CYCLE_LEN = 3
MAX_CYCLE_LEN = 6
MAX_CYCLE_STEP_WINDOW = 72


@dataclass
class Candidate:
    candidate_id: str
    method: str
    members: list[str]
    evidence: dict
    suspicion_score: float
    sender_balance_mismatch_count: int | None = None


_SHARED_DEVICE_QUERY: LiteralString = """
MATCH (a:Account)
WHERE a.device_id IS NOT NULL
WITH a.device_id AS device_id, collect(a) AS members
WHERE size(members) > 1
UNWIND members AS member
OPTIONAL MATCH (member)-[r:TRANSACTED_WITH]->(dest:Account)
RETURN device_id, [a IN members | a.account_id] AS accounts,
       collect(CASE WHEN r.step IS NOT NULL THEN
           {sender: member.account_id, recipient: dest.account_id, step: r.step} END) AS transactions
"""


def detect_shared_device(session: Session) -> list[Candidate]:
    """Use common recipients and repeat activity to corroborate a shared device."""
    candidates = []
    for row in session.run(_SHARED_DEVICE_QUERY):
        accounts = sorted(row["accounts"])
        transactions = row["transactions"]
        recipients = defaultdict(list)
        for tx in transactions:
            recipients[tx["recipient"]].append(tx)

        shared_recipients = 0
        peak_senders = 0
        repeat_senders = 0
        for transfers in recipients.values():
            by_sender = defaultdict(set)
            for tx in transfers:
                by_sender[tx["sender"]].add(tx["step"])
            if len(by_sender) >= 2:
                shared_recipients += 1
                peak_senders = max(peak_senders, len(by_sender))
                # A separate episode must be at least 12 hours apart.
                repeat_senders = max(repeat_senders, sum(max(s) - min(s) >= 12 for s in by_sender.values()))

        active_hours = len({tx["step"] for tx in transactions})
        frequency = len(transactions) / max(1, active_hours)
        score = min(1.0, 0.2 + 0.3 * peak_senders / len(accounts)
                    + 0.35 * repeat_senders / len(accounts) + 0.15 * min(1.0, frequency / 3))
        candidates.append(Candidate(
            candidate_id=f"shared_device::{row['device_id']}",
            method="shared_device",
            members=accounts,
            evidence={
                "device_id": row["device_id"], "size": len(accounts),
                "transaction_count": len(transactions), "active_hours": active_hours,
                "transactions_per_active_hour": round(frequency, 4),
                "shared_recipients": shared_recipients, "peak_senders_to_same_recipient": peak_senders,
                "repeat_senders_to_same_recipient": repeat_senders,
            },
            suspicion_score=round(score, 4),
        ))
    return candidates


_SHARED_PAYOUT_QUERY: LiteralString = """
MATCH (a:Account)-[r:TRANSACTED_WITH]->(dest:Account)
WHERE dest.kind = 'C' AND r.type = 'TRANSFER' AND r.step IS NOT NULL AND a <> dest
WITH dest, a, r ORDER BY r.step, a.account_id
WITH dest, collect({account_id: a.account_id, step: r.step}) AS transfers,
     count(DISTINCT a) AS unique_senders
WHERE unique_senders >= $min_size
RETURN dest.account_id AS dest, transfers
"""


def _payout_window(transfers: list[dict], min_size: int, max_size: int, max_window: int) -> tuple | None:
    """Prefer windows covering more distinct senders, then stronger behavioral evidence."""
    counts: Counter[str] = Counter()
    left = 0
    repeated = 0
    best = None
    best_rank = None
    for right, transfer in enumerate(transfers):
        sender = transfer["account_id"]
        counts[sender] += 1
        repeated += counts[sender] == 2
        while (transfer["step"] - transfers[left]["step"] > max_window
               or len(counts) > max_size):
            previous = transfers[left]["account_id"]
            repeated -= counts[previous] == 2
            counts[previous] -= 1
            if not counts[previous]:
                del counts[previous]
            left += 1
        unique = len(counts)
        if unique < min_size:
            continue
        span = transfer["step"] - transfers[left]["step"]
        total = right - left + 1
        frequency = total / (span + 1)
        score = (0.25 * unique / max_size + 0.3 / (1 + span / 12)
                 + 0.25 * repeated / unique + 0.2 * min(1.0, frequency))
        rank = (unique, score, -span)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = (sorted(counts), span, total, repeated, frequency, score)
    return best


def detect_shared_payout(
    session: Session,
    min_size: int = MIN_FANIN_SIZE,
    max_size: int = MAX_FANIN_SIZE,
    max_window: int = MAX_FANIN_STEP_WINDOW,
) -> list[Candidate]:
    """Measure distinct senders, transaction frequency, and repeat transfers within a window."""
    if not 1 <= min_size <= max_size or max_window < 0:
        raise ValueError("Invalid payout sender bounds or time window.")
    candidates = []
    for row in session.run(_SHARED_PAYOUT_QUERY, min_size=min_size):
        window = _payout_window(row["transfers"], min_size, max_size, max_window)
        if window is None:
            continue
        accounts, span, total, repeated, frequency, score = window
        candidates.append(Candidate(
            candidate_id=f"shared_payout::{row['dest']}",
            method="shared_payout",
            members=accounts,
            evidence={
                "payout_account": row["dest"], "size": len(accounts), "unique_senders": len(accounts),
                "step_window": span, "transaction_count": total, "repeat_senders": repeated,
                "transactions_per_hour": round(frequency, 4),
            },
            suspicion_score=round(score, 4),
        ))
    return candidates


_CIRCULAR_FLOW_QUERY: LiteralString = """
MATCH p = (a:Account)-[:TRANSACTED_WITH*3..6]->(a)
WITH nodes(p)[..-1] AS ns, relationships(p) AS rels
WHERE all(n IN ns WHERE single(member IN ns WHERE member = n))
  AND all(r IN rels WHERE r.type = 'TRANSFER' AND r.step IS NOT NULL)
WITH ns, rels, [r IN rels | r.step] AS steps
WHERE last(steps) - head(steps) <= $max_window
  AND all(i IN range(1, size(steps) - 1) WHERE steps[i - 1] <= steps[i])
  AND (head(steps) <> last(steps) OR all(n IN ns WHERE ns[0].account_id <= n.account_id))
RETURN [n IN ns | n.account_id] AS cycle_nodes, [r IN rels | r.amount] AS amounts, steps,
       [r IN rels | [r.oldbalanceOrg, r.newbalanceOrig]] AS sender_balances
"""


def _canonical_cycle(members: list[str]) -> tuple[str, ...]:
    """Give different starting points on the same directed cycle a common key."""
    start = members.index(min(members))
    return tuple(members[start:] + members[:start])


def _sender_balance_mismatch_count(
    amounts: tuple[float, ...], balances: list[list[float | None]] | None,
) -> int | None:
    """Check the representative lap when every sender balance is available."""
    if balances is None or len(balances) != len(amounts):
        return None
    mismatches = 0
    for amount, (old, new) in zip(amounts, balances):
        if old is None or new is None or not all(map(math.isfinite, (old, new, amount))):
            return None
        # PaySim amounts and balances are loaded as float32 before conversion to Python floats.
        tolerance = max(0.01, max(abs(old), abs(new), abs(amount)) * 2**-23)
        mismatches += not math.isclose(old - new, amount, rel_tol=0.0, abs_tol=tolerance)
    return mismatches


def _decay_consistency(amounts: tuple[float, ...]) -> float:
    """Measure how consistently each successive transfer retains a fraction of the previous one."""
    ratios = []
    for previous, current in pairwise(amounts):
        if previous <= 0 or current < 0:
            return 0.0
        ratios.append(current / previous)
    mean_ratio = statistics.mean(ratios) if ratios else 0.0
    if mean_ratio == 0.0:
        return 0.0
    non_increasing_fraction = sum(ratio <= 1.0 + 1e-9 for ratio in ratios) / len(ratios)
    ratio_cv = statistics.pstdev(ratios) / mean_ratio
    return non_increasing_fraction * max(0.0, 1 - ratio_cv)


def detect_circular_flow(
    session: Session, min_len: int = MIN_CYCLE_LEN, max_len: int = MAX_CYCLE_LEN,
    max_window: int = MAX_CYCLE_STEP_WINDOW,
) -> list[Candidate]:
    """Score chronological cycles using amount spread, per-hop decay and timing, and repeats."""
    if not 2 <= min_len <= max_len or max_window < 0:
        raise ValueError("Invalid cycle length bounds or time window.")
    query = cast(LiteralString, _CIRCULAR_FLOW_QUERY.replace("*3..6", f"*{min_len}..{max_len}"))
    episodes = defaultdict(dict)
    for row in session.run(query, max_window=max_window):
        key = _canonical_cycle(row["cycle_nodes"])
        path = (tuple(row["steps"]), tuple(row["amounts"]), tuple(row["cycle_nodes"]))
        episodes[key][path] = row.get("sender_balances")

    candidates = []
    for key, paths in episodes.items():
        # Earliest-finish interval scheduling counts distinct laps, avoiding combinations of the same legs.
        end = -1
        completed = []
        for steps, amounts, nodes in sorted(paths, key=lambda p: (p[0][-1], -p[0][0], p)):
            if steps[0] > end:
                completed.append((steps, amounts, nodes))
                end = steps[-1]

        steps, amounts, nodes = min(completed, key=lambda p: (p[0][-1] - p[0][0], p))
        length = len(key)
        span = steps[-1] - steps[0]
        avg_gap = span / (length - 1)
        mean_amount = statistics.mean(amounts)
        cv = statistics.pstdev(amounts) / mean_amount if mean_amount > 0 else 1.0
        repeats = len(completed)
        decay_consistency = _decay_consistency(amounts)
        score = (0.3 * max(0.0, 1 - cv) + 0.2 / (1 + avg_gap / 5)
                 + 0.2 * min(1.0, (repeats - 1) / 2) + 0.3 * decay_consistency)
        mismatches = _sender_balance_mismatch_count(amounts, paths[(steps, amounts, nodes)])
        candidates.append(Candidate(
            candidate_id=f"circular_flow::{'-'.join(key)}",
            method="circular_flow",
            members=list(key),
            evidence={
                "length": length, "amounts": list(amounts), "amount_cv": round(cv, 4),
                "decay_consistency": round(decay_consistency, 4),
                "chronological_accounts": list(nodes), "steps": list(steps), "step_window": span,
                "avg_step_gap": round(avg_gap, 4),
                "cycle_occurrences": repeats, "transaction_count": repeats * length,
                "sender_balance_mismatch_count": mismatches,
            },
            suspicion_score=round(score, 4),
            sender_balance_mismatch_count=mismatches,
        ))
    return candidates


def detect_candidates(
    driver: Driver,
    config: Neo4jConfig,
    min_fanin_size: int = MIN_FANIN_SIZE,
    max_fanin_size: int = MAX_FANIN_SIZE,
    max_fanin_step_window: int = MAX_FANIN_STEP_WINDOW,
    min_cycle_len: int = MIN_CYCLE_LEN,
    max_cycle_len: int = MAX_CYCLE_LEN,
    max_cycle_step_window: int = MAX_CYCLE_STEP_WINDOW,
) -> list[Candidate]:
    """Collect raw candidates for the downstream input guardrail."""
    with driver.session(database=config.database) as session:
        candidates = []
        candidates += detect_shared_device(session)
        candidates += detect_shared_payout(session, min_size=min_fanin_size, max_size=max_fanin_size, max_window=max_fanin_step_window)
        candidates += detect_circular_flow(session, min_len=min_cycle_len, max_len=max_cycle_len, max_window=max_cycle_step_window)
    return candidates


def evaluate_against_ground_truth(driver: Driver, config: Neo4jConfig, candidates: list[Candidate]) -> None:
    """Report held-out ring coverage using the evaluation's matching policy."""
    from src.eval.metrics import compute_ring_metrics, fetch_ring_tags, label_candidates

    tags = fetch_ring_tags(driver, config)
    coverage = compute_ring_metrics(tags, label_candidates(candidates, tags), [])
    print(f"\nHeld-out detector coverage: {coverage.detected}/{coverage.total} positive rings")


def summarize(candidates: list[Candidate]) -> None:
    print(f"candidates found: {len(candidates)}")
    by_method: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_method.setdefault(c.method, []).append(c)
    for method, group in sorted(by_method.items()):
        scores = sorted((c.suspicion_score for c in group), reverse=True)
        print(f"  {method:<15} n={len(group):<6} top scores={scores[:5]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-fanin-size", type=int, default=MIN_FANIN_SIZE)
    parser.add_argument("--max-fanin-size", type=int, default=MAX_FANIN_SIZE)
    parser.add_argument("--max-fanin-step-window", type=int, default=MAX_FANIN_STEP_WINDOW)
    parser.add_argument("--min-cycle-len", type=int, default=MIN_CYCLE_LEN)
    parser.add_argument("--max-cycle-len", type=int, default=MAX_CYCLE_LEN)
    parser.add_argument("--max-cycle-step-window", type=int, default=MAX_CYCLE_STEP_WINDOW)
    parser.add_argument("--check-ground-truth", action="store_true", help="run the developer recall spot-check after detection")
    args = parser.parse_args(argv)

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
            max_cycle_step_window=args.max_cycle_step_window,
        )
        summarize(candidates)
        if args.check_ground_truth:
            evaluate_against_ground_truth(driver, config, candidates)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
