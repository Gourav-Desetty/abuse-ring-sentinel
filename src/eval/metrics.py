"""Calibrate on training cohorts and evaluate held-out candidates and rings."""

from __future__ import annotations

import argparse
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
from src.data.plant_rings import BENIGN_METHODS, METHODS
from src.detection.detection import Candidate, detect_candidates

RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"

MAIN_EVAL_RUNS = 5 
STABILITY_RUNS = 6
STABILITY_CANDIDATES = 4

# Aggregate outcome when all runs for a candidate fail.
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
    sender_balance_mismatch_count: int | None = None


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
class RingMetrics:
    total: int
    detected: int
    flagged: int
    detection_recall: float | None
    verdict_recall: float | None


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
class MethodCalibration:
    floor: float | None
    n_positives: int
    n_negatives: int


@dataclasses.dataclass
class EvalResult:
    records: list[CandidateRecord]
    metrics: Metrics
    breakdown: list[BreakdownRow]
    stability: list[StabilityResult]
    n_candidates_total: int
    n_excluded_train: int
    n_eligible: int
    # Retain these candidates in records so coverage can be reported.
    n_unscored_failed: int = 0
    # Totals include the stability experiment.
    n_runs_attempted: int = 0
    n_runs_failed: int = 0
    rings: RingMetrics | None = None
    held_out_negatives: dict[str, int] = dataclasses.field(default_factory=dict)
    n_unmatched: int = 0
    n_duplicate_test: int = 0
    flag_calibration: dict[str, MethodCalibration] = dataclasses.field(default_factory=dict)


def fetch_ring_tags(driver, config: Neo4jConfig) -> dict[str, dict[str, Any]]:
    """Load the ground-truth tag stored on each account in Neo4j."""
    with driver.session(database=config.database) as session:
        rows = session.run(
            "MATCH (a:Account) WHERE a.ring_id IS NOT NULL "
            "RETURN a.account_id AS account_id, a.ring_id AS ring_id, a.ring_type AS ring_type, "
            "a.difficulty AS difficulty, a.split AS split, a.payout_account AS payout_account"
        )
        return {
            r["account_id"]: {key: r[key] for key in ("ring_id", "ring_type", "difficulty", "split", "payout_account")}
            for r in rows
        }


LabeledCandidates = list[tuple[Candidate, dict[str, Any]]]


def label_candidates(candidates: list[Candidate], ring_tags: dict[str, dict[str, Any]]) -> LabeledCandidates:
    """Require one same-method cohort and full member coverage, including the payout recipient."""
    members_by_ring = defaultdict(set)
    for account, tag in ring_tags.items():
        members_by_ring[tag["ring_id"]].add(account)

    labeled = []
    for candidate in candidates:
        accounts = set(candidate.members)
        recipient = candidate.evidence.get("payout_account")
        if candidate.method == "shared_payout" and recipient:
            accounts.add(recipient)
        tags = {ring_tags[a]["ring_id"]: ring_tags[a] for a in accounts if a in ring_tags}
        label = dict.fromkeys(("ring_id", "ring_type", "difficulty", "split", "ground_truth"))
        label["eligible"] = False
        if len(tags) == 1:
            matched, = tags.values()
            ring_type = matched["ring_type"]
            members = members_by_ring[matched["ring_id"]]
            if (
                (ring_type in METHODS or ring_type in BENIGN_METHODS)
                and BENIGN_METHODS.get(ring_type, ring_type) == candidate.method
                and len(members) >= 2 and members <= accounts
                and len(members) * 2 > len(accounts)
                and (candidate.method != "shared_payout" or (recipient and recipient == matched.get("payout_account")))
            ):
                label = {
                    **matched,
                    "ground_truth": "negative" if ring_type in BENIGN_METHODS else "positive",
                    "eligible": matched["split"] == "test",
                }
        labeled.append((candidate, label))
    return labeled


def label_candidate(candidate: Candidate, ring_tags: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return label_candidates([candidate], ring_tags)[0][1]


def independent_matches(labeled: LabeledCandidates, split: str) -> LabeledCandidates:
    """Keep one representative per cohort, selected before agent execution."""
    matches = {}
    for candidate, label in sorted(labeled, key=lambda item: (-item[0].suspicion_score, item[0].candidate_id)):
        if label["split"] == split:
            matches.setdefault(label["ring_id"], (candidate, label))
    return list(matches.values())


def calibrate_flag_floors(labeled: LabeledCandidates) -> dict[str, MethodCalibration]:
    """Minimize balanced classification error separately for each detector."""
    scores = {method: {"positive": [], "negative": []} for method in METHODS}
    for candidate, label in independent_matches(labeled, "train"):
        scores[candidate.method][label["ground_truth"]].append(candidate.suspicion_score)

    calibration = {}
    for method, groups in scores.items():
        positives, negatives = groups["positive"], groups["negative"]
        floor = None
        if positives and negatives:
            thresholds = {*positives, *negatives, 1.0001}
            floor = min(thresholds, key=lambda t: (
                sum(s < t for s in positives) * len(negatives)
                + sum(s >= t for s in negatives) * len(positives), t,
            ))
        calibration[method] = MethodCalibration(floor, len(positives), len(negatives))
    return calibration


def compute_ring_metrics(
    ring_tags: dict[str, dict[str, Any]], labeled: LabeledCandidates, records: list[CandidateRecord],
) -> RingMetrics:
    positive_ids = {
        tag["ring_id"] for tag in ring_tags.values()
        if tag["split"] == "test" and tag["ring_type"] in METHODS
    }
    detected = {label["ring_id"] for _, label in labeled} & positive_ids
    flagged = {record.ring_id for record in records if record.predicted_verdict == "ring_flagged"} & detected
    total = len(positive_ids)
    return RingMetrics(total, len(detected), len(flagged),
                       len(detected) / total if total else None, len(flagged) / total if total else None)


def scoreable(records: list[CandidateRecord]) -> list[CandidateRecord]:
    """Return candidates with at least one completed verdict."""
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


def select_borderline(records: list[CandidateRecord], k: int = STABILITY_CANDIDATES, *,
                      floor: float = FLAG_SCORE_FLOOR["shared_payout"]) -> list[CandidateRecord]:
    """Sample payout candidates near their calibrated flag floor."""
    pool = [r for r in records if r.method == "shared_payout"]
    pool.sort(key=lambda r: abs(r.suspicion_score - floor))
    return pool[:k]


def _run_one(app, candidate: Candidate, thread_id: str) -> dict[str, Any]:
    """Record escaped graph errors so the evaluation batch can continue."""
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
    """Return the modal verdict and fraction of completed runs that disagree."""
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
    """Vote across successful runs using a fresh checkpoint thread for each run."""
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


def run_stability_experiment(app, records: list[CandidateRecord], candidates_by_id: dict[str, Candidate], n_runs: int = STABILITY_RUNS, *, floor: float = FLAG_SCORE_FLOOR["shared_payout"]) -> list[StabilityResult]:
    if not n_runs:
        return []
    results = []
    for record in select_borderline(records, floor=floor):
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
    if n_main_runs < 1 or n_stability_runs < 0:
        raise ValueError("Evaluation requires at least one main run and zero or more stability runs.")
    neo4j_config = Neo4jConfig.from_env()
    driver = get_neo4j_driver(neo4j_config)
    try:
        candidates = detect_candidates(driver, neo4j_config)
        ring_tags = fetch_ring_tags(driver, neo4j_config)
    finally:
        driver.close()

    labeled = label_candidates(candidates, ring_tags)
    calibration = calibrate_flag_floors(labeled)
    flag_score_floors = {method: fit.floor for method, fit in calibration.items() if fit.floor is not None}
    test = independent_matches(labeled, "test")
    negatives = {
        tag["ring_id"]: tag["ring_type"] for tag in ring_tags.values()
        if tag["split"] == "test" and tag["ring_type"] in BENIGN_METHODS
    }
    if not test:
        raise RuntimeError("No held-out benchmark candidates matched. Rebuild the graph with the current cohort generator.")
    llm = build_llm(GroqConfig.from_env())
    guardrails = Guardrails(llm)
    app = build_graph(llm, guardrails, flag_score_floor=flag_score_floors)

    records: list[CandidateRecord] = []
    candidates_by_id: dict[str, Candidate] = {}
    for candidate, label in test:
        candidates_by_id[candidate.candidate_id] = candidate
        run = run_candidate_repeated(app, candidate, thread_id_prefix=f"eval-{candidate.candidate_id}", n_runs=n_main_runs)
        # Save a diagnosis supporting the modal verdict, or the first failure as a fallback.
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
                sender_balance_mismatch_count=candidate.sender_balance_mismatch_count,
            )
        )

    metrics = compute_metrics(records)
    breakdown = compute_breakdown(records)
    payout_floor = flag_score_floors.get("shared_payout")
    stability = (
        run_stability_experiment(app, records, candidates_by_id, n_runs=n_stability_runs, floor=payout_floor)
        if payout_floor is not None else []
    )

    return EvalResult(
        records=records,
        metrics=metrics,
        breakdown=breakdown,
        stability=stability,
        n_candidates_total=len(candidates),
        n_excluded_train=sum(label["split"] == "train" for _, label in labeled),
        n_eligible=len(records),
        n_unscored_failed=sum(1 for r in records if r.predicted_verdict == NO_VERDICT),
        n_runs_attempted=sum(r.n_runs for r in records) + sum(s.n_runs for s in stability),
        n_runs_failed=sum(r.n_failed for r in records) + sum(s.n_failed for s in stability),
        rings=compute_ring_metrics(ring_tags, labeled, records),
        held_out_negatives=dict(Counter(negatives.values())),
        n_unmatched=sum(label["ring_id"] is None for _, label in labeled),
        n_duplicate_test=sum(label["split"] == "test" for _, label in labeled) - len(test),
        flag_calibration=calibration,
    )


def save(result: EvalResult, path: Path = RESULTS_PATH) -> None:
    path.write_text(json.dumps(dataclasses.asdict(result), indent=2, default=str), encoding="utf-8")


def load(path: Path = RESULTS_PATH) -> EvalResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["records"] = [CandidateRecord(**r) for r in data["records"]]
    data["metrics"] = Metrics(**data["metrics"])
    data["breakdown"] = [BreakdownRow(**b) for b in data["breakdown"]]
    data["stability"] = [StabilityResult(**s) for s in data["stability"]]
    if data.get("rings") is not None:
        data["rings"] = RingMetrics(**data["rings"])
    calibration = data.get("flag_calibration", {})
    if "payout_floor" in data:
        calibration["shared_payout"] = {
            "floor": data.pop("payout_floor"),
            "n_positives": data.pop("payout_calibration_positives", 0),
            "n_negatives": data.pop("payout_calibration_negatives", 0),
        }
    data["flag_calibration"] = {method: MethodCalibration(**fit) for method, fit in calibration.items()}
    return EvalResult(**data)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=MAIN_EVAL_RUNS)
    parser.add_argument("--stability-runs", type=int, default=STABILITY_RUNS)
    args = parser.parse_args(argv)
    result = run_evaluation(args.runs, args.stability_runs)
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
    print(result.rings)


if __name__ == "__main__":
    main()
