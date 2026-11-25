"""Check that the agent returns needs_more_data for the unplanted PaySim payout candidate C680344850."""

from __future__ import annotations
import dataclasses
import sys
from src.agent.graph_state import build_graph, run_agent
from src.agent.guardrails import Guardrails
from src.agent.nodes import GroqConfig, build_llm
from src.data.build_graph import Neo4jConfig, get_driver
from src.detection.detection import detect_shared_payout

TARGET_PAYOUT_ACCOUNT = "C680344850"


def main() -> None:
    neo4j_config = Neo4jConfig.from_env()
    driver = get_driver(neo4j_config)
    try:
        with driver.session(database=neo4j_config.database) as session:
            candidates = detect_shared_payout(session)
    finally:
        driver.close()

    match = next((c for c in candidates if c.evidence.get("payout_account") == TARGET_PAYOUT_ACCOUNT), None)
    if match is None:
        print(f"{TARGET_PAYOUT_ACCOUNT} not found among current shared_payout candidates -- graph state may differ from the writeup.")
        sys.exit(1)

    candidate = dataclasses.asdict(match)
    print("candidate:")
    for k, v in candidate.items():
        print(f"  {k}: {v}")

    llm = build_llm(GroqConfig.from_env())
    guardrails = Guardrails(llm)
    app = build_graph(llm, guardrails)

    result = run_agent(app, candidate, thread_id=f"smoke-test-{TARGET_PAYOUT_ACCOUNT}")

    print("\n--- diagnosis ---")
    print(result.get("diagnosis"))
    print("\n--- output guardrail ---")
    print("is_grounded:", result.get("is_grounded"))
    print("hallucinated_claims:", result.get("hallucinated_claims"))
    print("\n--- final verdict ---")
    print("final_verdict:", result.get("final_verdict"))
    print("reason:", result.get("reason"))

    if result.get("run_failed"):
        # Distinct from a FAIL: nothing was proved either way, because the
        # LLM call never returned an answer (usually Groq's daily quota).
        print(f"\nINCONCLUSIVE: agent run failed during {result.get('failure_stage')} -- {result.get('failure_error')}")
        print("No verdict was produced; re-run once Groq quota allows.")
        sys.exit(2)

    if result.get("final_verdict") == "needs_more_data":
        print("\nPASS: guardrail correctly declined to flag on weak evidence.")
    else:
        print(f"\nFAIL: expected needs_more_data, got {result.get('final_verdict')!r}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
