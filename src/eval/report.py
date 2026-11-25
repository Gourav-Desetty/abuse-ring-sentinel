"""Render held-out metrics and worked agent examples from saved evaluation results."""

from __future__ import annotations

import sys
from pathlib import Path

from src.eval.metrics import NO_VERDICT, RESULTS_PATH, CandidateRecord, EvalResult, load

REPORT_PATH = Path(__file__).resolve().parent / "report.md"


def _fmt(x: float | None) -> str:
    return f"{x:.2%}" if x is not None else "n/a (no positive/predicted cases)"


def _fmt_flip(x: float | None) -> str:
    return f"{x:.0%}" if x is not None else "n/a"


def _fmt_runs(n_succeeded: int, n_failed: int, n_runs: int) -> str:
    """Format completed versus attempted runs and highlight any failures."""
    if n_failed == 0:
        return f"{n_succeeded}/{n_runs}"
    return f"**{n_succeeded}/{n_runs}** :warning: {n_failed} failed"


def _fmt_verdict(verdict: str, n_runs: int) -> str:
    if verdict == NO_VERDICT:
        return f"**insufficient data** (all {n_runs} runs failed -- not scored)"
    return verdict


def _find_positive_example(records: list[CandidateRecord]) -> CandidateRecord | None:
    flagged = [r for r in records if r.ground_truth == "positive" and r.predicted_verdict == "ring_flagged"]
    obvious = [r for r in flagged if r.difficulty == "obvious"]
    return (obvious or flagged or [None])[0]


def _find_negative_example(records: list[CandidateRecord]) -> CandidateRecord | None:
    negatives = [
        r for r in records
        if r.ground_truth == "negative" and r.predicted_verdict != NO_VERDICT
    ]
    with_diagnosis = [r for r in negatives if r.diagnosis]
    return (with_diagnosis or negatives or [None])[0]


def _worked_example(title: str, record: CandidateRecord | None) -> str:
    if record is None:
        return f"### {title}\n\n_No qualifying candidate found in this run's eligible set._\n"

    lines = [
        f"### {title}",
        "",
        f"- **candidate_id**: `{record.candidate_id}`",
        f"- **method**: `{record.method}`  |  **suspicion_score**: `{record.suspicion_score}`",
        f"- **members**: {record.members}",
        f"- **evidence**: `{record.evidence}`",
        f"- **ground truth**: {record.ground_truth}"
        + (f" ({record.ring_type}/{record.difficulty}, ring_id={record.ring_id}, split={record.split})" if record.ring_id else " (unplanted PaySim candidate)"),
        "",
        "**Diagnosis:**",
        "",
        "\n".join(f"> {line.rstrip()}".rstrip() for line in (record.diagnosis or "_none_").splitlines()),
        "",
        f"**Output guardrail:** is_grounded=`{record.is_grounded}`, hallucinated_claims=`{record.hallucinated_claims}`",
        "",
        f"**Final verdict:** `{record.predicted_verdict}`\n\n**Reason:** {record.reason}",
        "",
    ]
    if record.n_failed:
        lines[-1:] = [
            f":warning: **{record.n_failed} of {record.n_runs} runs failed** (LLM call never returned) and were "
            f"excluded from this candidate's vote: `{record.failure_errors}`",
            "",
        ]
    return "\n".join(lines)


def generate_report(result: EvalResult) -> str:
    m = result.metrics
    parts: list[str] = []

    parts.append("# Abuse Ring Sentinel -- Evaluation Report\n")
    if result.rings is None:
        parts.append(
            "> Historical evaluation. The original seven-candidate test also included a payout "
            "case used to set the flag threshold. [Current protocol](../../docs/EVALUATION.md)\n"
        )
    else:
        parts.append(
            "Candidate metrics use one representative per matched held-out cohort. "
            "Unmatched detections have unknown truth. Ring recall includes all held-out positive rings.\n"
        )
    parts.append(
        f"- Candidates from detection.py: **{result.n_candidates_total}**\n"
        f"- Excluded (matched a train-split ring): **{result.n_excluded_train}**\n"
        f"- Eligible: **{result.n_eligible}**"
        + (f" (of which **{result.n_unscored_failed}** unscored -- every run failed)" if result.n_unscored_failed else "")
        + "\n"
        f"- Agent runs: **{result.n_runs_attempted}** attempted, **{result.n_runs_failed}** failed\n"
    )
    if result.n_runs_failed:
        parts.append(
            f"> :warning: **Run health:** {result.n_runs_failed} of {result.n_runs_attempted} agent runs did not "
            "complete -- the LLM call exhausted its retries (typically Groq's daily token quota) and never returned "
            "a judgment. Those runs are excluded from every majority vote and flip rate below, and any candidate "
            "with no successful run at all is excluded from precision/recall entirely rather than counted as a "
            "declined flag. Check the `runs ok` column before reading any row as a stable result, and re-run the "
            "eval once quota allows for a clean measurement.\n"
        )
    if result.rings is not None:
        parts.append(
            f"- Unmatched detections excluded from candidate scoring: **{result.n_unmatched}**\n"
            f"- Duplicate held-out detections collapsed: **{result.n_duplicate_test}**"
        )
    for method, fit in sorted(result.flag_calibration.items()):
        floor = fit.floor if fit.floor is not None else "disabled (missing a calibration class)"
        parts.append(
            f"- `{method}` flag floor: **{floor}** "
            f"({fit.n_positives} positive / {fit.n_negatives} negative calibration cohorts)"
        )
    if result.rings is not None or result.flag_calibration:
        parts.append("")

    parts.append("## Overall metrics\n")
    if result.n_unscored_failed:
        parts.append(
            f"Computed over the {result.n_eligible - result.n_unscored_failed} eligible candidates that produced "
            f"at least one completed run; {result.n_unscored_failed} candidate(s) whose every run failed are "
            "excluded rather than counted as declined flags.\n"
        )
    parts.append("| Metric | Value |")
    parts.append("|---|---|")
    parts.append(f"| Candidate precision | {_fmt(m.precision)} |")
    parts.append(f"| Candidate recall | {_fmt(m.recall)} |")
    parts.append(f"| False-positive cost (count of negatives incorrectly flagged) | {m.fp_cost} |")
    parts.append(f"| True positives | {m.tp} |")
    parts.append(f"| False positives | {m.fp} |")
    parts.append(f"| False negatives | {m.fn} |")
    parts.append(f"| True negatives | {m.tn} |")
    parts.append("")

    if result.rings is not None:
        rings = result.rings
        parts.append("## Held-out cohort coverage\n")
        parts.append(f"- Ring detection recall: **{rings.detected}/{rings.total} ({_fmt(rings.detection_recall)})**")
        parts.append(f"- Ring verdict recall: **{rings.flagged}/{rings.total} ({_fmt(rings.verdict_recall)})**\n")
        parts.append("| Hard-negative cohort | Planted | Matched | Scored | Flagged |")
        parts.append("|---|---|---|---|---|")
        for kind, total in sorted(result.held_out_negatives.items()):
            matches = [r for r in result.records if r.ring_type == kind]
            scored = sum(r.predicted_verdict != NO_VERDICT for r in matches)
            flagged = sum(r.predicted_verdict == "ring_flagged" for r in matches)
            parts.append(f"| {kind} | {total} | {len(matches)} | {scored} | {flagged} |")
        parts.append("")

    parts.append("## Per-candidate verdicts\n")
    parts.append(
        "The scored verdict is the mode across completed runs. Flip rate measures disagreement "
        "with that mode. Read it alongside the number of completed runs.\n"
    )
    parts.append("| candidate_id | ground_truth | score | runs ok | verdicts (one per run) | modal verdict (scored) | flip rate |")
    parts.append("|---|---|---|---|---|---|---|")
    for r in result.records:
        parts.append(
            f"| `{r.candidate_id}` | {r.ground_truth} | {r.suspicion_score} | "
            f"{_fmt_runs(r.n_succeeded, r.n_failed, r.n_runs)} | {', '.join(r.verdicts)} | "
            f"{_fmt_verdict(r.predicted_verdict, r.n_runs)} | {_fmt_flip(r.flip_rate)} |"
        )
    parts.append("")

    parts.append("## Breakdown by ring_type / difficulty\n")
    parts.append("| Group | Kind | N | Flagged | Rate |")
    parts.append("|---|---|---|---|---|")
    for row in result.breakdown:
        rate_label = "recall" if row.kind == "positive" else "false-positive rate"
        parts.append(f"| {row.group} | {row.kind} | {row.n} | {row.n_flagged} | {row.rate:.2%} ({rate_label}) |")
    parts.append("")

    parts.append("## Verdict stability (side-experiment)\n")
    if result.stability:
        parts.append("| candidate_id | ground_truth | score | runs ok | verdicts (one per run) | modal verdict | flip rate |")
        parts.append("|---|---|---|---|---|---|---|")
        for s in result.stability:
            parts.append(
                f"| `{s.candidate_id}` | {s.ground_truth} | {s.suspicion_score} | "
                f"{_fmt_runs(s.n_succeeded, s.n_failed, s.n_runs)} | {', '.join(s.verdicts)} | "
                f"{_fmt_verdict(s.modal_verdict, s.n_runs)} | {_fmt_flip(s.flip_rate)} |"
            )
    else:
        parts.append("_No additional stability runs were recorded._")
    parts.append("")

    parts.append("## Worked examples\n")
    positive = _find_positive_example(result.records)
    if positive is None:
        positive = next((r for r in result.records if r.ground_truth == "positive" and r.diagnosis), None)
    parts.append(_worked_example("Held-out positive", positive))
    parts.append(_worked_example("Held-out negative", _find_negative_example(result.records)))

    return "\n".join(parts)


def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"{RESULTS_PATH} not found -- run `uv run python -m src.eval.metrics` first.")
        sys.exit(1)

    result = load()
    report = generate_report(result)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n(report also saved to {REPORT_PATH})")


if __name__ == "__main__":
    main()
