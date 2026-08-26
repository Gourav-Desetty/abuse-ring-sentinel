"""Render metrics.py's EvalResult into a human-readable markdown report --
overall precision/recall/false-positive cost, the ring_type/difficulty
breakdown, the verdict-stability side-experiment, and one worked example
each of a correctly-flagged true positive and a correctly-declined false
positive. This is the report the pitch draws from -- the false positive's
full evidence trail is kept visible, not just its number, since that worked
example is the strongest slide.

Run: uv run python -m src.eval.report   (loads metrics.py's saved results;
run metrics.py first if eval_results.json doesn't exist yet)
"""

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
    """"5/5" when every run completed; a flagged "2/5 !!" when some didn't --
    so a reader can never mistake repeated infrastructure failures for a
    stable, unanimous verdict.
    """
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
    unplanted = [r for r in records if r.ground_truth == "negative" and r.ring_id is None]
    preferred = [r for r in unplanted if r.evidence.get("payout_account") == "C680344850"]
    return (preferred or unplanted or [None])[0]


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
        + (f" ({record.ring_type}/{record.difficulty}, ring_id={record.ring_id}, split={record.split})" if record.ring_id else " (no planted ring -- real, unplanted PaySim data)"),
        "",
        "**Diagnosis:**",
        "",
        "> " + (record.diagnosis or "_none_").replace("\n", "\n> "),
        "",
        f"**Output guardrail:** is_grounded=`{record.is_grounded}`, hallucinated_claims=`{record.hallucinated_claims}`",
        "",
        f"**Final verdict:** `{record.predicted_verdict}`  \n**Reason:** {record.reason}",
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
    parts.append(
        "Scored on the held-out **test split only** -- the train split exists purely for threshold "
        "tuning. Candidates whose matched planted ring is in the train split are excluded from "
        "scoring entirely, not counted as errors either way.\n"
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
    parts.append(
        "> **Caveat:** `FLAG_SCORE_FLOOR` (the shared_payout evidence-sufficiency threshold in "
        "`agent/nodes.py`) was chosen during development by looking directly at this project's own "
        "C680344850 false positive and a few obvious/subtle score examples. That candidate's *outcome* "
        "is still legitimate to report -- it's real, unplanted data, never itself used for tuning -- but "
        "the threshold that routes it was not tuned blind. Treat the numbers below as representative of "
        "the design's behavior on a small synthetic set, not as a rigorously held-out benchmark.\n"
    )

    parts.append("## Overall metrics\n")
    if result.n_unscored_failed:
        parts.append(
            f"Computed over the {result.n_eligible - result.n_unscored_failed} eligible candidates that produced "
            f"at least one completed run; {result.n_unscored_failed} candidate(s) whose every run failed are "
            "excluded rather than counted as declined flags.\n"
        )
    parts.append("| Metric | Value |")
    parts.append("|---|---|")
    parts.append(f"| Precision | {_fmt(m.precision)} |")
    parts.append(f"| Recall | {_fmt(m.recall)} |")
    parts.append(f"| False-positive cost (count of negatives incorrectly flagged) | {m.fp_cost} |")
    parts.append(f"| True positives | {m.tp} |")
    parts.append(f"| False positives | {m.fp} |")
    parts.append(f"| False negatives | {m.fn} |")
    parts.append(f"| True negatives | {m.tn} |")
    parts.append("")

    n_runs_note = result.records[0].n_runs if result.records else "N"
    parts.append("## Per-candidate verdicts (majority vote)\n")
    parts.append(
        f"LLM sampling noise affects true-positive candidates too, not just deliberately borderline "
        f"ones -- two identical eval runs against the same graph produced different recall. So every "
        f"row above is a *majority vote*: each eligible candidate ran through the full agent "
        f"{n_runs_note} times independently, and its scored verdict is the modal (most common) one "
        f"across those runs. `flip rate` is the fraction of runs that disagreed with the modal "
        f"verdict for that candidate -- 0% means every run agreed.\n"
        f"\n`runs ok` is how many of the {n_runs_note} attempted runs actually completed. A run whose LLM call "
        f"exhausted its retries produced no judgment at all, shows as `run_failed`, and is dropped from both the "
        f"modal verdict and the flip rate -- so a row reading 0% flip on 2 successful runs is far weaker evidence "
        f"of stability than one reading 0% on {n_runs_note}.\n"
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
    parts.append(
        "A deeper look at the same phenomenon the majority vote above corrects for, but on the "
        "specific candidates closest to a real decision boundary (shared_payout's score floor) and "
        f"with more repeats ({result.stability[0].n_runs if result.stability else 'N'} vs. "
        f"{n_runs_note} above), to see how noisy the noisiest candidates actually are. Distinct from "
        "-- not folded into -- the per-candidate table above.\n"
    )
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
        parts.append("_No shared_payout candidates were eligible to sample borderline cases from._")
    parts.append("")

    parts.append("## Worked examples\n")
    parts.append(_worked_example("True positive, correctly flagged", _find_positive_example(result.records)))
    parts.append(_worked_example("False positive, correctly declined", _find_negative_example(result.records)))

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
