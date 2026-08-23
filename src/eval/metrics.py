"""Run the full agent pipeline over gds_detection.py's candidates and score
it against planted ground truth (precision, recall, false-positive cost, by
ring_type/difficulty), plus a verdict-stability side-experiment on
borderline candidates.

Every eligible candidate runs through the agent MAIN_EVAL_RUNS times (fresh
thread_id each time); its scored verdict is the modal one across those
runs, with per-candidate flip_rate reported alongside precision/recall.
Needed because two identical eval runs against the same graph produced
different recall (0.50 vs 0.167) -- LLM sampling noise affects even
clear-cut true positives, not just deliberately borderline ones.

Scoring is restricted to split="test" ring matches (never "train") --
`FLAG_SCORE_FLOOR` (agent/nodes.py) was tuned by eye against this project's
own C680344850 false positive, so that threshold wasn't tuned blind, and
the scored set still has to stay held-out. A candidate matching no planted
ring at all (`ring_id` is None) has no split to gate on and is always
eligible.

A run whose LLM call fails outright (e.g. Groq's daily quota) is tracked
via `RUN_FAILED_VERDICT` and dropped from the vote entirely -- missing
data, not evidence of agreement. A candidate whose every run fails is
excluded from precision/recall rather than scored as a declined flag.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.agent.graph_state import build_graph, run_agent
from src.agent.guardrails import Guardrails
from src.agent.nodes import FLAG_SCORE_FLOOR, RUN_FAILED_VERDICT, GroqConfig, build_llm
from src.data.build_graph import Neo4jConfig
from src.data.build_graph import get_driver as get_neo4j_driver
from src.detection.gds_detection import Candidate, detect_candidates

RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"

MAIN_EVAL_RUNS = 5
STABILITY_RUNS = 6
STABILITY_CANDIDATES = 4

# predicted_verdict for a candidate whose every run failed -- no verdict was
# ever produced, so it is neither flagged nor declined. Kept distinct from
# the three real verdicts (and from RUN_FAILED_VERDICT, which describes a
# single run rather than a candidate's aggregate outcome).
NO_VERDICT = "insufficient_data"


@dataclasses.dataclass
class CandidateRecord:
    candidate_id: str
    method: str
    suspicion_score: float
    members: list[str]
    evidence: dict[str, Any]
    ring_id: str | None
    ring_type: str | None  # the *matched* ring's type, distinct from `method`
    difficulty: str | None
    split: str | None
    ground_truth: str  # "positive" | "negative"
    predicted_verdict: str  # modal verdict across the *succeeded* runs, or NO_VERDICT
    n_runs: int  # runs attempted
    verdicts: list[str]  # one per attempted run; RUN_FAILED_VERDICT for failures
    flip_rate: float | None  # over succeeded runs only; None if none succeeded
    is_grounded: bool | None
    hallucinated_claims: list[str]
    diagnosis: str | None
    reason: str | None
    # Defaulted so results saved before failure-tracking existed still load.
    n_succeeded: int = 0
    n_failed: int = 0
    failure_errors: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Metrics:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float | None
    recall: float | None
    fp_cost: int  # == fp, under the more concrete "false-positive cost" name


@dataclasses.dataclass
class BreakdownRow:
    group: str
    kind: str  # "positive" | "negative"
    n: int
    n_flagged: int
    rate: float  # recall for positive groups, false-positive rate for negative groups


@dataclasses.dataclass
class StabilityResult:
    candidate_id: str
    ground_truth: str
    suspicion_score: float
    n_runs: int  # runs attempted
    verdicts: list[str]  # one per attempted run; RUN_FAILED_VERDICT for failures
    modal_verdict: str  # over succeeded runs only, or NO_VERDICT
    flip_rate: float | None  # over succeeded runs only; None if none succeeded
    n_succeeded: int = 0
    n_failed: int = 0


@dataclasses.dataclass
class EvalResult:
    records: list[CandidateRecord]
    metrics: Metrics
    breakdown: list[BreakdownRow]
    stability: list[StabilityResult]
    n_candidates_total: int
    n_excluded_train: int
    n_eligible: int
    # Eligible candidates every run of which failed -- present in `records`
    # but excluded from `metrics`/`breakdown` (no verdict was produced).
    n_unscored_failed: int = 0
    # Total individual agent runs attempted vs. failed, across records +
    # stability -- the headline "is this run trustworthy at all?" number.
    n_runs_attempted: int = 0
    n_runs_failed: int = 0


def fetch_ring_tags(driver, config: Neo4jConfig) -> dict[str, dict[str, Any]]:
    """account_id -> {ring_id, ring_type, difficulty, split} for every
    planted-ring member, straight from Neo4j's ground-truth properties.
    """
    with driver.session(database=config.database) as session:
        rows = session.run(
            "MATCH (a:Account) WHERE a.ring_id IS NOT NULL "
            "RETURN a.account_id AS account_id, a.ring_id AS ring_id, a.ring_type AS ring_type, "
            "a.difficulty AS difficulty, a.split AS split"
        )
        return {
            r["account_id"]: {"ring_id": r["ring_id"], "ring_type": r["ring_type"], "difficulty": r["difficulty"], "split": r["split"]}
            for r in rows
        }


def label_candidate(candidate: Candidate, ring_tags: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Determine a candidate's ground truth + eval eligibility (see module
    docstring). Matches on any member overlapping a tagged account -- ring
    planting doesn't reuse accounts across rings, so the first hit is the
    match.
    """
    matched = next((ring_tags[m] for m in candidate.members if m in ring_tags), None)

    if matched is None:
        return {"ring_id": None, "ring_type": None, "difficulty": None, "split": None, "ground_truth": "negative", "eligible": True}

    is_legit_cluster = matched["ring_type"] == "legit_cluster"
    return {
        "ring_id": matched["ring_id"],
        "ring_type": matched["ring_type"],
        "difficulty": matched["difficulty"],
        "split": matched["split"],
        "ground_truth": "negative" if is_legit_cluster else "positive",
        "eligible": matched["split"] == "test",
    }


def scoreable(records: list[CandidateRecord]) -> list[CandidateRecord]:
    """Records that actually have a verdict. A candidate whose every run
    failed produced none, and scoring it as a negative prediction would
    invent a judgment the agent never made.
    """
    return [r for r in records if r.predicted_verdict != NO_VERDICT]


def compute_metrics(records: list[CandidateRecord]) -> Metrics:
    records = scoreable(records)
    tp = sum(1 for r in records if r.ground_truth == "positive" and r.predicted_verdict == "ring_flagged")
    fn = sum(1 for r in records if r.ground_truth == "positive" and r.predicted_verdict != "ring_flagged")
    fp = sum(1 for r in records if r.ground_truth == "negative" and r.predicted_verdict == "ring_flagged")
    tn = sum(1 for r in records if r.ground_truth == "negative" and r.predicted_verdict != "ring_flagged")
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return Metrics(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, fp_cost=fp)


def compute_breakdown(records: list[CandidateRecord]) -> list[BreakdownRow]:
    groups: dict[str, list[CandidateRecord]] = defaultdict(list)
    for r in scoreable(records):
        key = f"{r.ring_type}/{r.difficulty}" if r.ground_truth == "positive" else (r.ring_type or "no_ring (coincidental)")
        groups[key].append(r)

    rows = []
    for key, recs in sorted(groups.items()):
        n_flagged = sum(1 for r in recs if r.predicted_verdict == "ring_flagged")
        rows.append(
            BreakdownRow(group=key, kind=recs[0].ground_truth, n=len(recs), n_flagged=n_flagged, rate=n_flagged / len(recs))
        )
    return rows


def select_borderline(records: list[CandidateRecord], k: int = STABILITY_CANDIDATES) -> list[CandidateRecord]:
    """Pick the k candidates closest to a real decision boundary. Only
    shared_payout has one (FLAG_SCORE_FLOOR) -- shared_device and
    circular_flow only ever score near 1.0 or don't appear as a candidate at
    all (see gds_detection.py's module docstring), so there's no genuine
    gray zone to test stability on outside shared_payout.
    """
    floor = FLAG_SCORE_FLOOR.get("shared_payout", 0.0)
    pool = [r for r in records if r.method == "shared_payout"]
    pool.sort(key=lambda r: abs(r.suspicion_score - floor))
    return pool[:k]


def _run_one(app, candidate: Candidate, thread_id: str) -> dict[str, Any]:
    """One agent run. Any exception that escapes the graph is caught here
    and turned into a failed-run result rather than aborting the batch: a
    single Groq outage must not cost every other candidate's results.
    """
    try:
        return run_agent(app, dataclasses.asdict(candidate), thread_id=thread_id)
    except Exception as err:  # noqa: BLE001 -- deliberately broad; see docstring
        return {
            "final_verdict": RUN_FAILED_VERDICT,
            "run_failed": True,
            "failure_stage": "run_agent",
            "failure_error": f"{type(err).__name__}: {err}",
            "reason": f"agent run raised before producing a verdict: {err}",
        }


def _run_succeeded(result: dict[str, Any]) -> bool:
    return not result.get("run_failed") and result.get("final_verdict") != RUN_FAILED_VERDICT


def _majority_vote(verdicts: list[str]) -> tuple[str, float | None]:
    """(modal verdict, flip_rate) over *completed* runs only -- flip_rate is
    the fraction of `verdicts` that disagreed with the modal one. An empty
    list means no run ever produced a verdict: (NO_VERDICT, None), never a
    silent fallback onto one of the real verdicts.
    """
    if not verdicts:
        return NO_VERDICT, None
    counts = Counter(verdicts)
    modal_verdict, modal_count = counts.most_common(1)[0]
    return modal_verdict, (len(verdicts) - modal_count) / len(verdicts)


@dataclasses.dataclass
class RepeatedRun:
    """Outcome of running one candidate n_runs times."""

    results: list[dict[str, Any]]  # every attempted run, failed ones included
    modal_verdict: str  # over succeeded runs only, or NO_VERDICT
    flip_rate: float | None  # over succeeded runs only, or None
    verdicts: list[str]  # one per attempted run, in order
    n_succeeded: int
    n_failed: int
    failure_errors: list[str]

    @property
    def succeeded_results(self) -> list[dict[str, Any]]:
        return [r for r in self.results if _run_succeeded(r)]


def run_candidate_repeated(app, candidate: Candidate, thread_id_prefix: str, n_runs: int) -> RepeatedRun:
    """Run `candidate` through the agent n_runs times (fresh thread_id each
    time, so no checkpoint is reused across runs).

    Runs whose LLM calls failed are recorded but EXCLUDED from the majority
    vote and flip_rate -- see this module's "Failed runs are not votes".
    """
    results = [_run_one(app, candidate, thread_id=f"{thread_id_prefix}-{i}") for i in range(n_runs)]
    succeeded = [r for r in results if _run_succeeded(r)]
    failed = [r for r in results if not _run_succeeded(r)]
    modal_verdict, flip_rate = _majority_vote([r["final_verdict"] for r in succeeded])
    return RepeatedRun(
        results=results,
        modal_verdict=modal_verdict,
        flip_rate=flip_rate,
        verdicts=[r["final_verdict"] if _run_succeeded(r) else RUN_FAILED_VERDICT for r in results],
        n_succeeded=len(succeeded),
        n_failed=len(failed),
        failure_errors=[str(r.get("failure_error") or r.get("reason")) for r in failed],
    )


def run_stability_experiment(app, records: list[CandidateRecord], candidates_by_id: dict[str, Candidate], n_runs: int = STABILITY_RUNS) -> list[StabilityResult]:
    results = []
    for record in select_borderline(records):
        candidate = candidates_by_id[record.candidate_id]
        run = run_candidate_repeated(app, candidate, thread_id_prefix=f"stability-{record.candidate_id}", n_runs=n_runs)
        results.append(
            StabilityResult(
                candidate_id=record.candidate_id,
                ground_truth=record.ground_truth,
                suspicion_score=record.suspicion_score,
                n_runs=n_runs,
                verdicts=run.verdicts,
                modal_verdict=run.modal_verdict,
                flip_rate=run.flip_rate,
                n_succeeded=run.n_succeeded,
                n_failed=run.n_failed,
            )
        )
    return results


def run_evaluation(n_main_runs: int = MAIN_EVAL_RUNS, n_stability_runs: int = STABILITY_RUNS) -> EvalResult:
    neo4j_config = Neo4jConfig.from_env()
    driver = get_neo4j_driver(neo4j_config)
    try:
        candidates = detect_candidates(driver, neo4j_config)
        ring_tags = fetch_ring_tags(driver, neo4j_config)
    finally:
        driver.close()

    llm = build_llm(GroqConfig.from_env())
    guardrails = Guardrails(llm)
    app = build_graph(llm, guardrails)

    records: list[CandidateRecord] = []
    candidates_by_id: dict[str, Candidate] = {}
    n_excluded_train = 0

    for candidate in candidates:
        label = label_candidate(candidate, ring_tags)
        if not label["eligible"]:
            n_excluded_train += 1
            continue

        candidates_by_id[candidate.candidate_id] = candidate
        run = run_candidate_repeated(app, candidate, thread_id_prefix=f"eval-{candidate.candidate_id}", n_runs=n_main_runs)
        # Representative run for the stored diagnosis/grounding trail: the
        # first *succeeded* run whose own verdict matches the modal verdict,
        # so what's displayed is consistent with the verdict actually used
        # for scoring. If every run failed there is no such run -- fall back
        # to the first attempt so the failure trail is still recorded.
        representative = next(
            (r for r in run.succeeded_results if r["final_verdict"] == run.modal_verdict), run.results[0]
        )
        records.append(
            CandidateRecord(
                candidate_id=candidate.candidate_id,
                method=candidate.method,
                suspicion_score=candidate.suspicion_score,
                members=candidate.members,
                evidence=candidate.evidence,
                ring_id=label["ring_id"],
                ring_type=label["ring_type"],
                difficulty=label["difficulty"],
                split=label["split"],
                ground_truth=label["ground_truth"],
                predicted_verdict=run.modal_verdict,
                n_runs=n_main_runs,
                verdicts=run.verdicts,
                flip_rate=run.flip_rate,
                is_grounded=representative.get("is_grounded"),
                hallucinated_claims=representative.get("hallucinated_claims") or [],
                diagnosis=representative.get("diagnosis"),
                reason=representative.get("reason"),
                n_succeeded=run.n_succeeded,
                n_failed=run.n_failed,
                failure_errors=run.failure_errors,
            )
        )

    metrics = compute_metrics(records)
    breakdown = compute_breakdown(records)
    stability = run_stability_experiment(app, records, candidates_by_id, n_runs=n_stability_runs)

    return EvalResult(
        records=records,
        metrics=metrics,
        breakdown=breakdown,
        stability=stability,
        n_candidates_total=len(candidates),
        n_excluded_train=n_excluded_train,
        n_eligible=len(records),
        n_unscored_failed=sum(1 for r in records if r.predicted_verdict == NO_VERDICT),
        n_runs_attempted=sum(r.n_runs for r in records) + sum(s.n_runs for s in stability),
        n_runs_failed=sum(r.n_failed for r in records) + sum(s.n_failed for s in stability),
    )


def save(result: EvalResult, path: Path = RESULTS_PATH) -> None:
    path.write_text(json.dumps(dataclasses.asdict(result), indent=2, default=str), encoding="utf-8")


def load(path: Path = RESULTS_PATH) -> EvalResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["records"] = [CandidateRecord(**r) for r in data["records"]]
    data["metrics"] = Metrics(**data["metrics"])
    data["breakdown"] = [BreakdownRow(**b) for b in data["breakdown"]]
    data["stability"] = [StabilityResult(**s) for s in data["stability"]]
    return EvalResult(**data)


def main() -> None:
    result = run_evaluation()
    save(result)
    print(f"saved results to {RESULTS_PATH}")
    print(f"total candidates: {result.n_candidates_total}, excluded (train-split match): {result.n_excluded_train}, eligible: {result.n_eligible}")
    print(f"agent runs: {result.n_runs_attempted} attempted, {result.n_runs_failed} failed (LLM call never returned an answer)")
    if result.n_runs_failed:
        print(
            f"WARNING: {result.n_runs_failed} run(s) failed and were excluded from the majority vote; "
            f"{result.n_unscored_failed} candidate(s) had no successful run at all and are unscored. "
            "These numbers are not a clean measurement -- re-run once the Groq quota resets."
        )
    print(result.metrics)


if __name__ == "__main__":
    main()
