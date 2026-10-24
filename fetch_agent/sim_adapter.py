"""
Thin adapter bridging natural-language scenario dicts to the existing
AegisCoord headless runner.  Returns a plain dict; never touches the network.
"""

import os
import sys
from typing import Dict, Any, List, Optional

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.runner import run_episode


# ── public interface ─────────────────────────────────────────────────────

def run_scenario(
    num_agents: int = 3,
    num_victims: int = 6,
    modes: Optional[List[str]] = None,
    steps: int = 40,
    seed: int = 42,
    grid_size: int = 10,
) -> Dict[str, Any]:
    """
    Run the existing headless simulation for each requested mode and
    return a combined results dict.

    Parameters mirror the existing run_episode() signature so we change
    nothing in env/ or backend/.
    """
    if modes is None:
        modes = ["greedy", "coordinated", "llm"]

    # Clamp to sane ranges
    num_agents = max(1, min(num_agents, 6))
    num_victims = max(1, min(num_victims, 15))
    steps = max(5, min(steps, 300))
    grid_size = max(5, min(grid_size, 20))

    results: Dict[str, Any] = {}

    for mode in modes:
        mode = mode.lower().strip()
        if mode not in ("greedy", "coordinated", "llm"):
            continue
        try:
            ep = run_episode(
                mode=mode,
                steps=steps,
                seed=seed,
                grid_size=grid_size,
                num_agents=num_agents,
                num_victims=num_victims,
                verbose=False,
            )
            results[mode] = ep["summary"]
        except Exception as exc:
            results[mode] = {"error": str(exc)}

    return results
