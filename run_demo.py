"""
Headless CLI Demo Runner for Disaster LLM Coordinator Platform.
Executes multi-agent disaster relief operations and outputs formatted metrics.
"""

import os
import sys
import argparse
import json

# Ensure local module priority
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.runner import run_episode


def main():
    parser = argparse.ArgumentParser(
        description="AegisCoord Headless CLI Simulation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="coordinated",
        choices=["greedy", "coordinated", "llm"],
        help="Coordination architecture: greedy (decentralized), coordinated (Hungarian), or llm (cognitive broker).",
    )
    parser.add_argument("--steps", type=int, default=40, help="Number of simulation steps to execute.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for world generation.")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size (N x N).")
    parser.add_argument("--agents", type=int, default=3, help="Number of relief agents.")
    parser.add_argument("--victims", type=int, default=8, help="Number of initial victims.")
    parser.add_argument("--export", type=str, default="trace.json", help="Path to export trace JSON file.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose step logs.")

    args = parser.parse_args()

    print("=" * 65)
    print(f"AEGISCOORD // HEADLESS DEMO RUNNER [{args.mode.upper()} MODE]")
    print(f"Grid: {args.grid_size}x{args.grid_size} | Fleet: {args.agents} Agents | Seed: {args.seed} | Steps: {args.steps}")
    print("=" * 65)

    result = run_episode(
        mode=args.mode,
        steps=args.steps,
        seed=args.seed,
        grid_size=args.grid_size,
        num_agents=args.agents,
        num_victims=args.victims,
        export_path=args.export,
        verbose=not args.quiet,
    )

    summary = result["summary"]
    print("\n" + "=" * 65)
    print("EPISODE COMPLETION SUMMARY")
    print("=" * 65)
    print(f"  Mode:                  {summary['mode'].upper()}")
    print(f"  Total Steps:           {summary['steps']}")
    print(f"  Total Rescued:         {summary['total_rescued']}")
    print(f"  Total Casualties:      {summary['total_died']}")
    total_handled = summary["total_rescued"] + summary["total_died"]
    survival_pct = (summary["total_rescued"] / total_handled * 100.0) if total_handled > 0 else 100.0
    print(f"  Survival Rate:         {survival_pct:.1f}%")
    print(f"  Avg Wait Time:         {summary['avg_wait_time']:.2f} s")
    print(f"  Jain's Fairness (J):   {summary['fairness_index']:.4f}")
    print(f"  Agent Workloads:       {summary['agent_rescues']}")
    print(f"  Execution Time:        {summary['elapsed_seconds']:.3f} s")
    print(f"  Trace File Export:     {os.path.abspath(args.export)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
