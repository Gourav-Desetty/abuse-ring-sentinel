# Evaluation protocol

## Calibration and held-out test

The seeded benchmark contains 90 positive rings and 45 known benign cohorts. Of those,
30 positives and 15 hard negatives are held out. The split is stratified by scenario
and difficulty, and no labeled account appears in both sets.

A candidate matches a cohort when it covers every cohort member, those members are
a majority of its accounts, and the detector matches the planted scenario. Payout
matches also require the labeled recipient. Candidates touching multiple cohorts
are excluded. The highest-scoring match represents each cohort, with ties resolved
by candidate ID before any model calls. Unmatched detections have unknown ground
truth and are counted in the report.

Each detector's flag floor minimizes balanced classification error on the scores
of its matched calibration cohorts. A method needs both positive and benign
calibration cohorts to get a floor. Evaluation applies these floors to grounded
positive verdicts and fits them exclusively on the training split.

## Metrics

- **Candidate precision** measures how many flagged, scored candidates are positives.
  **Candidate recall** measures the fraction of scored positive candidates flagged.
- **Ring detection recall** divides the number of matched positive test rings by all
  positive test rings. **Ring verdict recall** uses the rings whose candidate was
  flagged as the numerator. Both denominators include rings that never yielded a candidate.
- **Hard-negative coverage** shows how many examples of each benign scenario were
  planted, matched, scored, and flagged.
- **Run coverage** records model failures. Successful runs vote for the final verdict.
  A candidate with zero completed attempts is excluded from candidate metrics. Its
  ring remains in the ring-level denominator. The optional stability experiment
  repeats selected payout candidates near the calibrated floor.

The evaluation estimates performance on planted scenarios in PaySim. It measures
candidate precision on the labeled cohort, so production precision remains unknown.
See [the demo](DEMO.md) for a reproducible run and [the full report](../src/eval/report.md)
for evidence, model diagnoses, and final verdicts.

## Development notes

- Before repeated voting, evaluation runs against the same graph yielded 50%
  and about 17% recall. The default evaluation uses five attempts per candidate
  and six additional attempts for up to four borderline payout candidates.
- The verifier uses Groq structured output with `method="json_schema"`.
  The tool-calling path returned HTTP 400 when the model omitted the requested
  tool call. The JSON schema path returned the expected `ValidationResult`.
- The grounding check covers concrete facts such as IDs, amounts, and time
  windows. The diagnosis provides the interpretation and severity rating.
- Groq's daily token cap can outlast retry backoff. The evaluator records the
  resulting missing verdict so run coverage is visible in the report.
