"""
Headless Episode Runner & Trace Exporter for Disaster Multi-Agent Simulations.
Executes episodes across greedy, coordinated (Hungarian), and LLM modes,
exporting execution traces and summary analytics.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List, Optional

# Ensure local module priority
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import DisasterReliefEnv
from agents import GreedyAgent, TaskExecutor, MultiAgentController
from backend.coordinator.simple_coordinator import SimpleCoordinator
from coordinator.llm_coordinator import LLMCoordinator


def run_episode(
    mode: str = "coordinated",
    steps: int = 40,
    seed: int = 42,
    grid_size: int = 10,
    num_agents: int = 3,
    num_victims: int = 8,
    num_obstacles: int = 12,
    export_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes a complete headless simulation episode and returns trace and metrics.
    """
    env = DisasterReliefEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        max_victims=num_victims,
        seed=seed,
    )
    obs = env.reset()

    mode = mode.lower()
    if mode == "greedy":
        controller = MultiAgentController([GreedyAgent(i) for i in range(num_agents)])
        coordinator = None
        executor = None
    elif mode == "coordinated":
        controller = None
        coordinator = SimpleCoordinator(assign_interval=1, urgency_weight=4.0)
        executor = TaskExecutor(num_agents=num_agents)
    elif mode == "llm":
        controller = None
        coordinator = LLMCoordinator()
        executor = TaskExecutor(num_agents=num_agents)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'greedy', 'coordinated', or 'llm'.")

    trace: List[Dict[str, Any]] = []
    start_time = time.time()

    for s in range(steps):
        obs = env.get_observation()
        assignments = {}

        if mode == "greedy":
            actions = controller.act(obs)
            assignments = {str(i): "greedy_nearest" for i in range(num_agents)}
        else:
            raw_assignments = coordinator.assign(obs, step=s)
            for aid_key, task in raw_assignments.items():
                aid = int(aid_key)
                assignments[str(aid)] = task
                if task is not None and executor._targets[aid] is None:
                    executor.set_assignment(aid, task)
            actions = executor.act(obs)

        next_obs, rewards, done, info = env.step(actions)

        step_record = {
            "step": s,
            "actions": [int(a) for a in actions],
            "assignments": assignments,
            "total_rescued": int(env.total_rescued),
            "total_died": int(env.total_died),
            "active_victims": len([v for v in env.victims if not v.rescued and not v.died]),
        }
        trace.append(step_record)

        if verbose and (s % 10 == 0 or s == steps - 1):
            print(
                f"[Step {s:02d}/{steps}] Rescued: {env.total_rescued} | "
                f"Casualties: {env.total_died} | "
                f"Active: {step_record['active_victims']}"
            )

    elapsed = time.time() - start_time

    rescued_victims = [v for v in env.victims if v.rescued]
    avg_wait = float(sum(v.wait_time for v in rescued_victims) / len(rescued_victims)) if rescued_victims else 0.0

    # Jain's fairness index across agent rescues
    rescues = [float(a.total_rescues) for a in env.agents]
    total_r = sum(rescues)
    if total_r == 0:
        fairness = 1.0
    else:
        n = len(rescues)
        fairness = float((total_r ** 2) / (n * sum(r * r for r in rescues)))

    summary = {
        "mode": mode,
        "seed": seed,
        "steps": steps,
        "total_rescued": int(env.total_rescued),
        "total_died": int(env.total_died),
        "avg_wait_time": round(avg_wait, 2),
        "fairness_index": round(fairness, 4),
        "agent_rescues": [int(a.total_rescues) for a in env.agents],
        "agent_moves": [int(a.total_moves) for a in env.agents],
        "elapsed_seconds": round(elapsed, 4),
    }

    if export_path:
        os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "trace": trace}, f, indent=2)
        if verbose:
            print(f"Trace saved to {export_path}")

    return {"summary": summary, "trace": trace}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless Episode Runner")
    parser.add_argument("--mode", type=str, default="coordinated", choices=["greedy", "coordinated", "llm"])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export", type=str, default=None)
    args = parser.parse_args()

    result = run_episode(
        mode=args.mode,
        steps=args.steps,
        seed=args.seed,
        export_path=args.export,
        verbose=True,
    )
    print("\nSimulation Finished:")
    print(json.dumps(result["summary"], indent=2))
