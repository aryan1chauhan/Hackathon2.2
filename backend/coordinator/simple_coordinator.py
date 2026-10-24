"""
Classical Combinatorial Hungarian Algorithm Coordinator for Multi-Agent Disaster Relief.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from agents.greedy_agent import manhattan_distance
from agents.pathfinding import a_star_search
from env.constants import STATE_CARRYING, CellType


def _extract_obstacles(obs: Dict[str, Any]) -> set[Tuple[int, int]]:
    obstacles = set()
    grid_rows = obs.get("grid", [])
    for r_idx, row in enumerate(grid_rows):
        if isinstance(row, str):
            cells = row.split()
            for c_idx, val in enumerate(cells):
                if val == str(int(CellType.OBSTACLE)):
                    obstacles.add((c_idx, r_idx))
    return obstacles


class SimpleCoordinator:
    """
    Global fleet coordinator using the Hungarian algorithm to compute an optimal,
    bipartite assignment minimizing fleet transit time while prioritizing critical victims.
    """

    def __init__(self, assign_interval: int = 1, urgency_weight: float = 4.0):
        self.assign_interval = assign_interval
        self.urgency_weight = urgency_weight
        self.current_assignments: Dict[int, Optional[str]] = {}
        self.last_assigned_step = -1

    def assign(self, env_or_obs: Any, step: int = 0) -> Dict[int, Optional[str]]:
        """
        Computes optimal global assignment mapping agent_id -> 'rescue V{id}' or None.
        Accepts either DisasterReliefEnv instance or observation dictionary.
        """
        # Periodic interval check
        if (
            self.last_assigned_step >= 0
            and step - self.last_assigned_step < self.assign_interval
            and self.current_assignments
        ):
            return self.current_assignments

        # Normalize observation
        if isinstance(env_or_obs, dict):
            obs = env_or_obs
        elif hasattr(env_or_obs, "get_observation"):
            obs = env_or_obs.get_observation()
        else:
            raise ValueError("env_or_obs must be DisasterReliefEnv or observation dict")

        agents_data = obs.get("agents", [])
        num_agents = len(agents_data)
        assignments: Dict[int, Optional[str]] = {a["id"]: None for a in agents_data}

        # Filter active, unrescued, non-dead victims
        active_victims = [
            v for v in obs.get("victims", [])
            if not v.get("rescued", False) and not v.get("died", False)
        ]

        # Edge Case EC-3.4: Zero active victims on the grid
        if not active_victims or num_agents == 0:
            self.current_assignments = assignments
            self.last_assigned_step = step
            return assignments

        # Build Cost Matrix: C[i, j] = True_Dist(Agent_i, Victim_j) - urgency_weight * Urgency(Victim_j)
        num_victims = len(active_victims)
        cost_matrix = np.zeros((num_agents, num_victims), dtype=np.float64)
        obstacles = _extract_obstacles(obs)
        grid_size = obs.get("grid_size", 10)

        for i, agent in enumerate(agents_data):
            agent_pos = tuple(agent["pos"])
            is_carrying = (
                agent.get("carrying") is not None
                or agent.get("state") == STATE_CARRYING
            )

            for j, victim in enumerate(active_victims):
                if is_carrying:
                    # Heavily penalize assigning an agent already carrying a victim
                    cost_matrix[i, j] = 1e6
                else:
                    v_pos = tuple(victim["pos"])
                    if agent_pos == v_pos:
                        dist = 0
                    else:
                        path = a_star_search(agent_pos, v_pos, obstacles, grid_size)
                        dist = len(path) if path else 1e6

                    if dist >= 1e6:
                        cost_matrix[i, j] = 1e6
                    else:
                        urgency = float(victim.get("urgency", 1.0))
                        # Lower cost means higher priority assignment
                        cost_matrix[i, j] = float(dist) - (self.urgency_weight * urgency)

        # Solve Hungarian Bipartite Matching in polynomial time O(N^3)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Populate assignments
        for r, c in zip(row_ind, col_ind):
            agent_id = agents_data[r]["id"]
            # Only assign if agent is not carrying and cost is valid
            if cost_matrix[r, c] < 1e5:
                victim_id = active_victims[c]["id"]
                assignments[agent_id] = f"rescue V{victim_id}"
            else:
                assignments[agent_id] = None

        self.current_assignments = assignments
        self.last_assigned_step = step
        return assignments
