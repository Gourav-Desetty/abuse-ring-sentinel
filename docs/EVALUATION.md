# Evaluation and Groq notes

## Scoring

- A candidate takes its label from the first member with a stored ring tag. Test-split collusion matches are positive, test-split `legit_cluster` matches are negative, and train-split matches are excluded. Candidates without a tag are eligible negatives.
- The full agent runs five times per candidate, using a fresh checkpoint thread each time. The most frequent verdict among successful runs is scored. The flip rate is the fraction of those runs that disagree with it.
- Failed calls are recorded and excluded from the vote. When all attempts fail, the candidate is unscored. The stability experiment runs six times per candidate for up to four shared-payout candidates closest to the `0.4` flag threshold.

The saved run scored seven of 25 detections, with six positives and one negative. It flagged three of the positives and zero negatives, yielding 100% precision and 50% recall. Treat those results as a description of this run. The unplanted PaySim negative, `C680344850`, helped inform the `0.4` payout flag threshold and also appears in the scored set. [Saved report](../src/eval/report.md)

## Development notes

- Before repeated voting, evaluation runs against the same graph returned 50% and about 17% recall. That variation motivated repeat runs for all eligible candidates.
- Payout fan-in size and timing can look similar for coincidental and planted groups. `C680344850` scored `0.2885` and received a grounded “Yes” diagnosis. The separate flag floor held it back at the cost of missing two planted payout candidates in the saved run.
- The output guardrail uses `with_structured_output(..., method="json_schema")` with Groq's `openai/gpt-oss-120b`. During development, the tool-calling method returned HTTP 400 with “Tool choice is required, but model did not call a tool.” The JSON schema path returned the expected `ValidationResult`.
- An earlier verifier treated severity ratings such as “Medium” as unsupported claims, reducing recall. The current verifier checks concrete facts such as IDs, amounts, and time windows. The diagnosis supplies the verdict, mechanism, and severity.
- Groq's daily token cap can outlast retry backoff. The guardrail fails closed and records the missing judgment so a quota failure is excluded from scoring. Rerun after the quota resets if it affected coverage.
