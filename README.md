# Abuse Ring Sentinel

Abuse Ring Sentinel finds potentially collusive accounts in a Neo4j transaction graph. Cypher queries look for shared devices, payout fan-in, and circular flows. A Groq model assesses each candidate, and a separate model call checks its factual claims against the graph evidence before the agent issues a verdict.

## Quickstart

Install Python 3.12+ and `uv`, set up a Neo4j database and a Groq API key, and place the PaySim CSV at `archive/paysim.csv`. Copy `.env.example` to `.env` and fill in the credentials and database name. [Dataset setup](docs/DATA.md)

```bash
uv sync
uv run python -X utf8 main.py
```

The pipeline loads PaySim, plants labeled rings, detects candidates, and writes an evaluation report. Graph loading clears the configured Neo4j database by default. Use `--skip-build` to evaluate an existing graph or `--help` for other options.

## Evaluation

The seeded benchmark holds out **30 positive rings and 15 hard negatives** across
shared devices, personal transfers to a common recipient, and circular flows. The
report includes candidate precision, ring detection and verdict recall, hard-negative
coverage, and worked examples with model diagnoses and verdicts.
[Evaluation protocol](docs/EVALUATION.md) · [Full report](src/eval/report.md)

The saved single-pass Groq run scored 44 candidates with **70.7% candidate
precision** and **96.7% ring verdict recall**. It flagged 12 of the 15 hard
negatives, including every benign shared-device and circular-flow example.

[Reproduce the demo and explore the graph in Neo4j Browser](docs/DEMO.md).

## Tests

Run the detection, guardrail, and evaluation tests locally. They use mocked Neo4j results and model responses and require no service credentials.

```bash
uv run python -m unittest discover -s tests -v
```

See the [architecture](ARCHITECTURE.md) for pipeline details and [dataset setup](docs/DATA.md) for how the benchmark rings were planted.
