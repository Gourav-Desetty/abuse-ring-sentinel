# Abuse Ring Sentinel

Detects collusion rings between accounts/merchants/devices that look
individually legitimate but form a fraud ring when viewed as a graph.

**Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager.** Defense-only.
Honest metrics, including false-positive cost, below.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for how it works.

## Quickstart

**Prerequisites**

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- A Neo4j instance (an [AuraDB Free](https://neo4j.com/product/auradb/) instance is enough — see the note on GDS in `ARCHITECTURE.md`)
- A [Groq API key](https://console.groq.com/keys)
- `archive/paysim.csv` — not included in this repo (too large); see
  [`docs/DATA.md`](docs/DATA.md) for the dataset source, the two Kaggle
  quirks that make it unsuitable as a ring-ground-truth label on its own,
  and exactly how synthetic ground truth was built on top of it

```bash
uv sync
cp .env.example .env   # then fill in NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD / GROQ_API_KEY
```

**Run the pipeline end to end**

```bash
uv run python main.py
```

Runs build graph → detect → eval → report in one command, with the same
defaults the results below were generated from. `--help` shows the few
overridable options (sample fraction, skip the build step to reuse an
already-loaded graph, etc.).

**Or run each step individually**, for more visibility into what each stage
produces:

```bash
# 1. Build the graph: load PaySim, plant synthetic collusion rings, push into Neo4j.
#    --sample-frac (not --nrows) so the sample spans PaySim's full 743-step
#    timeline -- see ARCHITECTURE.md for why that matters.
python -m src.data.build_graph --sample-frac 0.0157 --wipe

# 2. Pull candidate rings out of the graph (Cypher pattern detection, see ARCHITECTURE.md).
python -m src.detection.gds_detection --check-ground-truth

# 3. Prove the guardrail discriminates in both directions.
python -m src.agent.smoke_test            # real false positive -> needs_more_data
python -m src.agent.smoke_test_positive    # real planted ring   -> ring_flagged

# 4. Score the full agent pipeline against held-out ground truth and generate the report.
python -m src.eval.metrics
python -m src.eval.report
```

`src/data/load_paysim.py` and `src/data/plant_rings.py` can also be run standalone
(`uv run python -m src.data.load_paysim`) to inspect the loaded/planted data on their own.
`main.py` deliberately doesn't run the smoke tests (step 3 above) -- they're a proof
artifact to read the output of, not part of the scored pipeline.

## Results

Full report with the complete diagnosis/guardrail trail for every worked example:
[`src/eval/report.md`](src/eval/report.md).

Scored on the held-out **test split only** — 7 of 25 detected candidates were
eligible (18 matched a train-split ring and were excluded from scoring
entirely, not counted as errors either way). At this sample size every cell
below is either 0% or 100%; see the full report for the caveat on statistical
significance and on how `FLAG_SCORE_FLOOR` was chosen.

LLM sampling noise turned out to affect true-positive candidates, not just
deliberately borderline ones — two identical eval runs against the same
graph produced different recall. So each candidate's verdict below is a
**majority vote over 5 independent runs** through the full agent, not a
single sample; a `flip rate` > 0% means the runs didn't all agree. Runs
whose LLM call fails outright (e.g. Groq's daily quota) are tracked
separately and excluded from the vote rather than silently counted as a
verdict — this run had **0 of 53** agent calls fail, so the numbers below
are a clean measurement, not degraded by infrastructure noise. Full
per-candidate vote breakdown: [`src/eval/report.md`](src/eval/report.md).

| Metric | Value |
|---|---|
| Precision | 100.00% |
| Recall | 50.00% |
| False-positive cost | 0 |
| True positives | 3 |
| False positives | 0 |
| False negatives | 3 |
| True negatives | 1 |

| Group | Kind | N | Flagged | Rate |
|---|---|---|---|---|
| circular_flow/obvious | positive | 1 | 1 | 100.00% (recall) |
| circular_flow/subtle | positive | 1 | 1 | 100.00% (recall) |
| shared_device/obvious | positive | 1 | 1 | 100.00% (recall) |
| shared_device/subtle | positive | 1 | 0 | 0.00% (recall) |
| shared_payout/obvious | positive | 1 | 0 | 0.00% (recall) |
| shared_payout/subtle | positive | 1 | 0 | 0.00% (recall) |
| no_ring (coincidental) | negative | 1 | 0 | 0.00% (false-positive rate) |

Recall reflects 3 deliberate non-flags, not 3 misses of unknown cause — each
has a specific, inspectable reason: **2** (`shared_payout/obvious`,
`shared_payout/subtle`) held back by the precision-tuned score floor despite
a grounded "yes" diagnosis, and **1** (`shared_device/subtle`) where the
modal judgment across 5 runs was "unclear" rather than "yes" (2 of the 5
runs did say yes — `flip_rate 40%`, the noisiest candidate in this run).
Full diagnosis/guardrail trail for each is in
[`src/eval/report.md`](src/eval/report.md).

### Worked example: true positive, correctly flagged

`shared_device::DEV-shared_device-obvious-001` (suspicion_score `0.9`) — three
accounts sharing one `device_id`. Diagnosis: "Yes," citing the device_id and
member count. Output guardrail: `is_grounded=True`, no hallucinated claims.
**Final verdict: `ring_flagged`** (modal over 5 runs; `flip_rate 40%` — 2 of
5 runs landed on `needs_more_data` instead, a reminder that even a correct
majority verdict here isn't unanimous).

### Worked example: false positive, correctly declined

`shared_payout::C680344850` (suspicion_score `0.2885`) — three real,
*unplanted* PaySim accounts that coincidentally transferred into the same
downstream account within a 3-step window. The diagnosis said "Yes" and was
grounded (nothing was hallucinated — every cited field is real), but the
evidence was too thin: `suspicion_score 0.2885` is below the shared_payout
flag floor of `0.4`. **Final verdict: `needs_more_data`** (unanimous across
5 runs, `flip_rate 0%`), not a false flag.

This is the case worth reading in full in the report — it's real data the
detector genuinely found, not a strawman, and it's why the pipeline has both
a score-based guardrail *and* an LLM grounding check rather than relying on
either alone.

## Repository layout

```
src/
  data/
    load_paysim.py    # load + clean archive/paysim.csv
    plant_rings.py     # plant synthetic shared_device / shared_payout / circular_flow rings + legit_cluster hard negatives
    build_graph.py      # push accounts/transactions/ring ground truth into Neo4j
  detection/
    gds_detection.py    # candidate ring extraction (see ARCHITECTURE.md re: "GDS" in the name)
  agent/
    graph_state.py       # LangGraph wiring: detect -> diagnose -> verify -> decide
    nodes.py              # the four node functions + Groq call
    guardrails.py          # input (score floor) + output (grounding check) guardrails
    smoke_test.py            # proof: real false positive -> needs_more_data
    smoke_test_positive.py    # proof: real planted ring -> ring_flagged
  eval/
    metrics.py             # precision/recall/false-positive cost + verdict-stability side-experiment
    report.py               # renders eval_results.json into report.md
```
