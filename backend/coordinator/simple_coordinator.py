from typing import Dict, List, Tuple, Optional
from env import DisasterReliefEnv
import itertools

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    linear_sum_assignment = None
    _HAS_SCIPY = False


def _cost(agent_pos, victim, urgency_weight: float = 4.0) -> float:
    """Cost function for assigning agent -> victim.

    Lower cost is better. We combine travel distance (Manhattan) and urgency.
    We subtract urgency_weight * urgency_score so high-urgency victims reduce cost.
    """
    dist = abs(agent_pos[0] - victim.pos[0]) + abs(agent_pos[1] - victim.pos[1])
    urgency = victim.urgency()
    return dist - urgency_weight * urgency


class SimpleCoordinator:
    """Coordinator that assigns victims to agents using min-cost assignment.

    If SciPy is available it uses the Hungarian algorithm (`linear_sum_assignment`).
    Otherwise it falls back to a brute-force search over permutations limited
    to a small candidate set (top victims by urgency).
    """

    def __init__(self, assign_interval: int = 1, urgency_weight: float = 4.0):
        self.assign_interval = assign_interval
        self.last_assign_step = -1
        self.urgency_weight = urgency_weight

    def assign(self, env: DisasterReliefEnv, step: int) -> Dict[int, Optional[str]]:
        """Return mapping agent_id -> assignment string or None"""
        assignments: Dict[int, Optional[str]] = {i: None for i in range(env.num_agents)}

        # Only assign on the configured interval
        if (step - self.last_assign_step) < self.assign_interval:
            return assignments

        self.last_assign_step = step

        # Active victims
        active_victims = [v for v in env.victims if not v.rescued and not v.died]
        if not active_victims:
            return assignments

        # Sort victims by urgency descending and limit candidates
        victims_sorted = sorted(active_victims, key=lambda v: -v.urgency())
        idle_agents = [a for a in env.agents if a.is_available()]
        if not idle_agents:
            return assignments

        max_candidates = max(len(idle_agents) * 4, len(idle_agents))
        candidates = victims_sorted[:max_candidates]

        # If SciPy is available, build cost matrix and run Hungarian
        if _HAS_SCIPY and len(candidates) > 0:
            import numpy as np

            n_agents = len(idle_agents)
            n_cand = len(candidates)
            cost_mat = np.zeros((n_agents, n_cand), dtype=float)
            for i, agent in enumerate(idle_agents):
                for j, victim in enumerate(candidates):
                    cost_mat[i, j] = _cost(agent.pos, victim, self.urgency_weight)

            # Solve assignment (rows->agents, cols->victims)
            row_ind, col_ind = linear_sum_assignment(cost_mat)
            for r, c in zip(row_ind, col_ind):
                if r < len(idle_agents) and c < len(candidates):
                    aid = idle_agents[r].id
                    vid = candidates[c].id
                    assignments[aid] = f'rescue V{vid}'

            return assignments

        # Fallback: brute-force over permutations for small sizes
        n_agents = len(idle_agents)
        n_candidates = len(candidates)
        best_total = None
        best_mapping = None
        max_k = min(n_agents, n_candidates)
        if max_k == 0:
            return assignments
        for k in range(1, max_k + 1):
            for perm in itertools.permutations(range(n_candidates), k):
                total = 0.0
                mapping = {}
                for a_idx, victim_idx in enumerate(perm):
                    agent = idle_agents[a_idx]
                    victim = candidates[victim_idx]
                    total += _cost(agent.pos, victim, self.urgency_weight)
                    mapping[agent.id] = victim.id

                if best_total is None or total < best_total:
                    best_total = total
                    best_mapping = mapping

        if best_mapping:
            for aid, vid in best_mapping.items():
                assignments[aid] = f'rescue V{vid}'

        return assignments
