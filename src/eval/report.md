# Abuse Ring Sentinel -- Evaluation Report

Candidate metrics use one representative per matched held-out cohort. Unmatched detections have unknown truth. Ring recall includes all held-out positive rings.

- Candidates from detection.py: **180**
- Excluded (matched a train-split ring): **90**
- Eligible: **44**
- Agent runs: **44** attempted, **0** failed

- Unmatched detections excluded from candidate scoring: **46**
- Duplicate held-out detections collapsed: **0**
- `circular_flow` flag floor: **0.6147** (20 positive / 10 negative calibration cohorts)
- `shared_device` flag floor: **0.7833** (20 positive / 10 negative calibration cohorts)
- `shared_payout` flag floor: **0.4716** (20 positive / 10 negative calibration cohorts)

## Overall metrics

| Metric | Value |
|---|---|
| Candidate precision | 93.33% |
| Candidate recall | 96.55% |
| False-positive cost (count of negatives incorrectly flagged) | 2 |
| True positives | 28 |
| False positives | 2 |
| False negatives | 1 |
| True negatives | 13 |

## Held-out cohort coverage

- Ring detection recall: **29/30 (96.67%)**
- Ring verdict recall: **28/30 (93.33%)**

| Hard-negative cohort | Planted | Matched | Scored | Flagged |
|---|---|---|---|---|
| legit_circular_flow | 5 | 5 | 5 | 0 |
| legit_personal_transfers | 5 | 5 | 5 | 1 |
| legit_shared_device | 5 | 5 | 5 | 1 |

## Per-candidate verdicts

The scored verdict is the mode across completed runs. Flip rate measures disagreement with that mode. Read it alongside the number of completed runs.

| candidate_id | ground_truth | score | runs ok | verdicts (one per run) | modal verdict (scored) | flip rate |
|---|---|---|---|---|---|---|
| `circular_flow::C1067021698-C1766604058-C1639173389-C154790264` | positive | 0.9667 | 1/1 | ring_flagged | ring_flagged | 0% |
| `circular_flow::C1097116860-C982520341-C408531032-C426985154` | positive | 0.9667 | 1/1 | ring_flagged | ring_flagged | 0% |
| `circular_flow::C1116496219-C342798647-C1455500035-C1671757644-C694457946` | positive | 0.9667 | 1/1 | ring_flagged | ring_flagged | 0% |
| `circular_flow::C1305166224-C1978861206-C1722662191-C1945895458-C416275121` | positive | 0.9667 | 1/1 | ring_flagged | ring_flagged | 0% |
| `circular_flow::C1926028321-C598311871-C57199881` | positive | 0.9667 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_device::DEV-C286849878` | positive | 0.955 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_device::DEV-C1860176522` | positive | 0.9409 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_device::DEV-C1038712782` | positive | 0.9318 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_device::DEV-C1741841229` | positive | 0.9292 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_device::DEV-C1495433254` | positive | 0.9278 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_device::DEV-C860888887` | positive | 0.9077 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_device::DEV-C1239248847` | negative | 0.9071 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_device::DEV-C1856699617` | positive | 0.9056 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_device::DEV-C1537670287` | positive | 0.9 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_device::DEV-C820920634` | positive | 0.9 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_device::DEV-C2031618368` | positive | 0.8494 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_payout::C939168439` | positive | 0.736 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_payout::C1120408335` | positive | 0.7208 | 1/1 | ring_flagged | ring_flagged | 0% |
| `circular_flow::C1648846292-C1893583917-C2029549826` | positive | 0.6705 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_payout::C1408936980` | positive | 0.6646 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_payout::C269333852` | positive | 0.6646 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_payout::C1849951552` | positive | 0.6333 | 1/1 | ring_flagged | ring_flagged | 0% |
| `circular_flow::C2094115027-C819214262-C838864898-C267024038` | positive | 0.6309 | 1/1 | ring_flagged | ring_flagged | 0% |
| `circular_flow::C1190332895-C438568805-C610506976-C2104622371-C1192569632-C2043511178` | positive | 0.6199 | 1/1 | ring_flagged | ring_flagged | 0% |
| `circular_flow::C1150259607-C982663092-C1858113249-C590455494-C937613633-C782989602` | positive | 0.61 | 1/1 | needs_more_data | needs_more_data | 0% |
| `shared_payout::C508850074` | positive | 0.5843 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_payout::C295710973` | negative | 0.5751 | 1/1 | needs_more_data | needs_more_data | 0% |
| `shared_payout::C722270827` | negative | 0.5707 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_device::DEV-C1689130830` | negative | 0.5583 | 1/1 | needs_more_data | needs_more_data | 0% |
| `shared_payout::C1863409918` | positive | 0.5525 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_device::DEV-C1282343859` | negative | 0.55 | 1/1 | needs_more_data | needs_more_data | 0% |
| `shared_device::DEV-C1449446685` | negative | 0.55 | 1/1 | needs_more_data | needs_more_data | 0% |
| `shared_device::DEV-C556654108` | negative | 0.55 | 1/1 | needs_more_data | needs_more_data | 0% |
| `shared_payout::C806078563` | positive | 0.515 | 1/1 | ring_flagged | ring_flagged | 0% |
| `circular_flow::C1047049439-C737969496-C385182550-C1430870313-C1225660861-C1406783023` | negative | 0.5044 | 1/1 | needs_more_data | needs_more_data | 0% |
| `shared_payout::C506904979` | negative | 0.4838 | 1/1 | needs_more_data | needs_more_data | 0% |
| `shared_payout::C406489636` | positive | 0.4789 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_payout::C58711561` | positive | 0.4751 | 1/1 | ring_flagged | ring_flagged | 0% |
| `shared_payout::C674792551` | negative | 0.444 | 1/1 | needs_more_data | needs_more_data | 0% |
| `circular_flow::C1041448817-C1309300813-C1986623721-C492489765` | negative | 0.4277 | 1/1 | needs_more_data | needs_more_data | 0% |
| `circular_flow::C1295487286-C1358804387-C732697563-C1810237786` | negative | 0.4226 | 1/1 | needs_more_data | needs_more_data | 0% |
| `circular_flow::C1890983340-C747979499-C216386219-C765998194-C896782006` | negative | 0.3961 | 1/1 | needs_more_data | needs_more_data | 0% |
| `circular_flow::C1387906538-C894356972-C295743495-C2014600293` | negative | 0.3928 | 1/1 | needs_more_data | needs_more_data | 0% |
| `shared_payout::C493389696` | negative | 0.3511 | 1/1 | needs_more_data | needs_more_data | 0% |

## Breakdown by ring_type / difficulty

| Group | Kind | N | Flagged | Rate |
|---|---|---|---|---|
| circular_flow/obvious | positive | 5 | 5 | 100.00% (recall) |
| circular_flow/subtle | positive | 4 | 3 | 75.00% (recall) |
| legit_circular_flow | negative | 5 | 0 | 0.00% (false-positive rate) |
| legit_personal_transfers | negative | 5 | 1 | 20.00% (false-positive rate) |
| legit_shared_device | negative | 5 | 1 | 20.00% (false-positive rate) |
| shared_device/obvious | positive | 5 | 5 | 100.00% (recall) |
| shared_device/subtle | positive | 5 | 5 | 100.00% (recall) |
| shared_payout/obvious | positive | 5 | 5 | 100.00% (recall) |
| shared_payout/subtle | positive | 5 | 5 | 100.00% (recall) |

## Verdict stability (side-experiment)

_No additional stability runs were recorded._

## Worked examples

### Held-out positive

- **candidate_id**: `circular_flow::C1067021698-C1766604058-C1639173389-C154790264`
- **method**: `circular_flow`  |  **suspicion_score**: `0.9667`
- **members**: ['C1067021698', 'C1766604058', 'C1639173389', 'C154790264']
- **evidence**: `{'length': 4, 'amounts': [452358.78, 452358.78, 452358.78, 452358.78], 'amount_cv': 0.0, 'decay_consistency': 1.0, 'chronological_accounts': ['C1766604058', 'C1639173389', 'C154790264', 'C1067021698'], 'steps': [458, 459, 460, 461], 'step_window': 3, 'avg_step_gap': 1.0, 'cycle_occurrences': 3, 'transaction_count': 12, 'sender_balance_mismatch_count': None}`
- **ground truth**: positive (circular_flow/obvious, ring_id=circular_flow-obvious-072, split=test)

**Diagnosis:**

> **1. Likely collusion ring? – Yes**
>
> **2. Supporting evidence**
> - **Circular pattern**: `"length": 4` with `"chronological_accounts": ["C1766604058","C1639173389","C154790264","C1067021698"]` shows a closed loop of four accounts.
> - **Identical transaction amounts**: `"amounts": [452358.78,452358.78,452358.78,452358.78]` and `"amount_cv": 0.0` (zero coefficient of variation) indicate perfectly matched values across the loop.
> - **Consistent decay**: `"decay_consistency": 1.0` (no decay across hops) reinforces a deliberate, balanced flow.
> - **Tight timing**: `"steps": [458,459,460,461]`, `"step_window": 3`, and `"avg_step_gap": 1.0` show the transfers occurred in successive, adjacent steps, leaving little temporal dispersion.
> - **Repeated execution**: `"cycle_occurrences": 3` and `"transaction_count": 12` demonstrate the loop was run three times (4 × 3 = 12 transactions).
>
> **3. Plausible collusion mechanism – Circular flow**
> The evidence points to money being passed from one account to the next in a closed loop, repeatedly, with no loss or gain at each hop (perfect amount match and decay consistency). This matches the “circular_flow” detection method.
>
> **4. Overall suspicion severity – High**
> - Large uniform amount (≈ $452 k) per transaction.
> - Multiple exact repetitions (3 cycles, 12 transactions).
> - Perfect amount and timing consistency, which is atypical for legitimate activity.
>
> These factors together warrant a **High** severity rating.

**Output guardrail:** is_grounded=`True`, hallucinated_claims=`[]`

**Final verdict:** `ring_flagged`

**Reason:** diagnosis answered 'yes' to Q1, grounded=True

### Held-out negative

- **candidate_id**: `shared_device::DEV-C1239248847`
- **method**: `shared_device`  |  **suspicion_score**: `0.9071`
- **members**: ['C1239248847', 'C1776885369', 'C2087380320']
- **evidence**: `{'device_id': 'DEV-C1239248847', 'size': 3, 'transaction_count': 8, 'active_hours': 7, 'transactions_per_active_hour': 1.1429, 'shared_recipients': 1, 'peak_senders_to_same_recipient': 3, 'repeat_senders_to_same_recipient': 3}`
- **ground truth**: negative (legit_shared_device/None, ring_id=legit_shared_device-benign-105, split=test)

**Diagnosis:**

> **1. Likelihood of a collusion ring:** **Yes**
>
> **2. Supporting evidence (exact fields/values):**
> - **size:** 3 → three accounts are linked to the same device.
> - **shared_recipients:** 1 → all three accounts send to a common recipient.
> - **peak_senders_to_same_recipient:** 3 → the maximum number of distinct senders to that recipient is three (i.e., all members).
> - **repeat_senders_to_same_recipient:** 3 → the same three senders repeatedly target the same recipient.
> - **transaction_count:** 8 → multiple transactions are occurring through this shared device.
>
> **3. Plausible collusion mechanism:** **Shared‑device funnel** – the three accounts operate on a single device (as indicated by the detection method and the “device_id” field) and repeatedly send funds to the same recipient, suggesting a coordinated payout funnel.
>
> **4. Overall suspicion severity:** **High** – the combination of multiple accounts on one device, repeated fan‑in to a single recipient, and a high repeat‑sender count points to intentional, coordinated activity rather than incidental overlap.

**Output guardrail:** is_grounded=`True`, hallucinated_claims=`[]`

**Final verdict:** `ring_flagged`

**Reason:** diagnosis answered 'yes' to Q1, grounded=True
