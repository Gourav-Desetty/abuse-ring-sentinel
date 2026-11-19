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

The saved run scored **100% precision and 50% recall** on **seven candidates**. It flagged three of six planted positives and correctly withheld a flag from the only negative. Each verdict came from five independent agent runs. The single negative helped inform the payout flag threshold, so the precision figure is descriptive and needs independent validation. [Evaluation notes](docs/EVALUATION.md) · [Full report](src/eval/report.md)

## Tests

Run the detection, guardrail, and evaluation tests locally. They use mocked Neo4j results and model responses and require no service credentials.

```bash
uv run python -m unittest discover -s tests -v
```

See the [architecture](ARCHITECTURE.md) for pipeline details and [dataset setup](docs/DATA.md) for how the benchmark rings were planted.
