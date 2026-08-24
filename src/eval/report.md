# Abuse Ring Sentinel -- Evaluation Report

Scored on the held-out **test split only** -- the train split exists purely for threshold tuning. Candidates whose matched planted ring is in the train split are excluded from scoring entirely, not counted as errors either way.

- Candidates from gds_detection.py: **25**
- Excluded (matched a train-split ring): **18**
- Eligible: **7**
- Agent runs: **53** attempted, **0** failed

> **Caveat:** `FLAG_SCORE_FLOOR` (the shared_payout evidence-sufficiency threshold in `agent/nodes.py`) was chosen during development by looking directly at this project's own C680344850 false positive and a few obvious/subtle score examples. That candidate's *outcome* is still legitimate to report -- it's real, unplanted data, never itself used for tuning -- but the threshold that routes it was not tuned blind. Treat the numbers below as representative of the design's behavior on a small synthetic set, not as a rigorously held-out benchmark.

## Overall metrics

| Metric | Value |
|---|---|
| Precision | 100.00% |
| Recall | 50.00% |
| False-positive cost (count of negatives incorrectly flagged) | 0 |
| True positives | 3 |
| False positives | 0 |
| False negatives | 3 |
| True negatives | 1 |

## Per-candidate verdicts (majority vote)

LLM sampling noise affects true-positive candidates too, not just deliberately borderline ones -- two identical eval runs against the same graph produced different recall. So every row above is a *majority vote*: each eligible candidate ran through the full agent 5 times independently, and its scored verdict is the modal (most common) one across those runs. `flip rate` is the fraction of runs that disagreed with the modal verdict for that candidate -- 0% means every run agreed.

`runs ok` is how many of the 5 attempted runs actually completed. A run whose LLM call exhausted its retries produced no judgment at all, shows as `run_failed`, and is dropped from both the modal verdict and the flip rate -- so a row reading 0% flip on 2 successful runs is far weaker evidence of stability than one reading 0% on 5.

| candidate_id | ground_truth | score | runs ok | verdicts (one per run) | modal verdict (scored) | flip rate |
|---|---|---|---|---|---|---|
| `shared_device::DEV-shared_device-obvious-001` | positive | 0.9 | 5/5 | ring_flagged, needs_more_data, needs_more_data, ring_flagged, ring_flagged | ring_flagged | 40% |
| `shared_device::DEV-shared_device-subtle-005` | positive | 1.0 | 5/5 | ring_flagged, needs_more_data, needs_more_data, ring_flagged, ring_flagged | ring_flagged | 40% |
| `shared_payout::C680344850` | negative | 0.2885 | 5/5 | needs_more_data, needs_more_data, needs_more_data, needs_more_data, needs_more_data | needs_more_data | 0% |
| `shared_payout::C-PAYOUT-shared_payout-obvious-009` | positive | 0.3333 | 5/5 | needs_more_data, needs_more_data, needs_more_data, needs_more_data, needs_more_data | needs_more_data | 0% |
| `shared_payout::C-PAYOUT-shared_payout-subtle-015` | positive | 0.0694 | 5/5 | needs_more_data, needs_more_data, needs_more_data, needs_more_data, needs_more_data | needs_more_data | 0% |
| `circular_flow::C1120408335-C838513034-C950052162-C779752097` | positive | 1.0 | 5/5 | ring_flagged, ring_flagged, ring_flagged, ring_flagged, ring_flagged | ring_flagged | 0% |
| `circular_flow::C1303683796-C704279786-C1579909190-C269333852-C1532502819-C1913558175` | positive | 0.7988 | 5/5 | needs_more_data, needs_more_data, ring_flagged, needs_more_data, ring_flagged | needs_more_data | 40% |

## Breakdown by ring_type / difficulty

| Group | Kind | N | Flagged | Rate |
|---|---|---|---|---|
| circular_flow/obvious | positive | 1 | 1 | 100.00% (recall) |
| circular_flow/subtle | positive | 1 | 0 | 0.00% (recall) |
| no_ring (coincidental) | negative | 1 | 0 | 0.00% (false-positive rate) |
| shared_device/obvious | positive | 1 | 1 | 100.00% (recall) |
| shared_device/subtle | positive | 1 | 1 | 100.00% (recall) |
| shared_payout/obvious | positive | 1 | 0 | 0.00% (recall) |
| shared_payout/subtle | positive | 1 | 0 | 0.00% (recall) |

## Verdict stability (side-experiment)

A deeper look at the same phenomenon the majority vote above corrects for, but on the specific candidates closest to a real decision boundary (shared_payout's score floor) and with more repeats (6 vs. 5 above), to see how noisy the noisiest candidates actually are. Distinct from -- not folded into -- the per-candidate table above.

| candidate_id | ground_truth | score | runs ok | verdicts (one per run) | modal verdict | flip rate |
|---|---|---|---|---|---|---|
| `shared_payout::C-PAYOUT-shared_payout-obvious-009` | positive | 0.3333 | 6/6 | needs_more_data, needs_more_data, needs_more_data, needs_more_data, needs_more_data, needs_more_data | needs_more_data | 0% |
| `shared_payout::C680344850` | negative | 0.2885 | 6/6 | needs_more_data, needs_more_data, needs_more_data, needs_more_data, needs_more_data, needs_more_data | needs_more_data | 0% |
| `shared_payout::C-PAYOUT-shared_payout-subtle-015` | positive | 0.0694 | 6/6 | needs_more_data, needs_more_data, needs_more_data, needs_more_data, needs_more_data, needs_more_data | needs_more_data | 0% |

## Worked examples

### True positive, correctly flagged

- **candidate_id**: `shared_device::DEV-shared_device-obvious-001`
- **method**: `shared_device`  |  **suspicion_score**: `0.9`
- **members**: ['C1305361573', 'C1424878287', 'C266548472']
- **evidence**: `{'device_id': 'DEV-shared_device-obvious-001', 'size': 3}`
- **ground truth**: positive (shared_device/obvious, ring_id=shared_device-obvious-001, split=test)

**Diagnosis:**

> **1. Likely collusion ring?** Yes  
> 
> **2. Supporting evidence**  
> - `device_id: "DEV-shared_device-obvious-001"` – the same device is linked to all three members.  
> - `size: 3` – exactly three accounts are connected to that device.  
> 
> **3. Plausible collusion mechanism** Shared‑device ring (multiple accounts operating from a single device).  
> 
> **4. Overall suspicion severity** Medium** (three accounts tied to one device suggests coordinated activity, but without additional signals (e.g., transaction flow) the risk is not yet critical).**

**Output guardrail:** is_grounded=`True`, hallucinated_claims=`[]`

**Final verdict:** `ring_flagged`  
**Reason:** diagnosis answered 'yes' to Q1, grounded=True

### False positive, correctly declined

- **candidate_id**: `shared_payout::C680344850`
- **method**: `shared_payout`  |  **suspicion_score**: `0.2885`
- **members**: ['C1041448817', 'C1477009521', 'C223224253']
- **evidence**: `{'payout_account': 'C680344850', 'size': 3, 'step_window': 3}`
- **ground truth**: negative (no planted ring -- real, unplanted PaySim data)

**Diagnosis:**

> **1. Likely collusion ring?** Unclear – the only signal is a single shared payout account, which could be coincidental without additional corroboration.
> 
> **2. Supporting evidence**  
> - The three members are linked to the same payout account **`payout_account: "C680344850"`**.  
> - The graph reports **`size: 3`** (three members) and a **`step_window: 3`** (the time window in which the shared payouts were observed).
> 
> **3. Plausible collusion mechanism** Shared‑payout funnel – multiple accounts (C1041448817, C1477009521, C223224253) appear to route funds to a common payout account (C680344850).
> 
> **4. Overall suspicion severity** Unclear (insufficient evidence to assign a definitive severity level).

**Output guardrail:** is_grounded=`True`, hallucinated_claims=`[]`

**Final verdict:** `needs_more_data`  
**Reason:** diagnosis answered 'unclear' to Q1, grounded=True
