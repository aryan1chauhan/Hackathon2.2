"""
Offline smoke test for sim_adapter — no network, no uAgents, no API keys.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fetch_agent.sim_adapter import run_scenario


def main():
    print("=" * 60)
    print("sim_adapter local test — running 3 modes, 15 steps, seed 42")
    print("=" * 60)

    results = run_scenario(
        num_agents=2,
        num_victims=6,
        modes=["greedy", "coordinated", "llm"],
        steps=15,
        seed=42,
    )

    for mode, summary in results.items():
        if "error" in summary:
            print(f"\n[{mode.upper()}]  ERROR: {summary['error']}")
            continue
        rescued = summary["total_rescued"]
        died = summary["total_died"]
        handled = rescued + died
        survival = (rescued / handled * 100) if handled else 100.0
        print(
            f"\n[{mode.upper()}]"
            f"  rescued={rescued}  died={died}"
            f"  survival={survival:.1f}%"
            f"  avg_wait={summary['avg_wait_time']:.2f}"
            f"  fairness={summary['fairness_index']:.4f}"
        )

    # Basic assertions
    assert "coordinated" in results, "coordinated mode missing"
    assert "greedy" in results, "greedy mode missing"
    coord = results["coordinated"]
    assert "total_rescued" in coord, "summary keys missing"
    assert coord["total_rescued"] >= 0
    print("\n[PASS] All assertions passed.")


if __name__ == "__main__":
    main()
