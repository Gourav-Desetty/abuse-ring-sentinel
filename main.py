"""Single-command pipeline entrypoint: build graph -> detect -> eval -> report.
"""

from __future__ import annotations

import argparse

from src.data import build_graph
from src.detection import detection
from src.eval import metrics, report

DEFAULT_SAMPLE_FRAC = 0.0157


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=DEFAULT_SAMPLE_FRAC,
        help=f"fraction of PaySim rows to load (default {DEFAULT_SAMPLE_FRAC}, matching the documented results)",
    )
    parser.add_argument("--no-wipe", action="store_true", help="don't clear the existing graph before loading")
    parser.add_argument("--skip-build", action="store_true", help="skip the build-graph step (reuse an already-loaded graph)")
    args = parser.parse_args(argv)

    if args.skip_build:
        print("=== 1/4: build graph -- skipped (--skip-build) ===")
    else:
        print("=== 1/4: build graph (load PaySim, plant rings, push to Neo4j) ===")
        build_graph_argv = ["--sample-frac", str(args.sample_frac)]
        if not args.no_wipe:
            build_graph_argv.append("--wipe")
        build_graph.main(build_graph_argv)

    print("\n=== 2/4: detect candidate rings ===")
    detection.main(["--check-ground-truth"])

    print("\n=== 3/4: score the agent against held-out ground truth ===")
    metrics.main()

    print("\n=== 4/4: render the report ===")
    report.main()

    print("\nDone -- full report: src/eval/report.md")


if __name__ == "__main__":
    main()
