"""
Greedy heuristic multi-agent controller for disaster relief.
"""

from typing import List, Dict, Any, Tuple
from agents.agent_interface import AgentInterface
from env.constants import (
    ACTION_STAY,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_PICKUP,
    ACTION_DROPOFF,
    STATE_CARRYING,
)


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Computes Manhattan metric d(p1, p2) = |x1 - x2| + |y1 - y2|."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def step_towards(src: Tuple[int, int], dst: Tuple[int, int]) -> int:
    """Returns the single-step cardinal action that reduces distance to dst."""
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]

    if dx > 0:
        return ACTION_RIGHT
    elif dx < 0:
        return ACTION_LEFT
    elif dy > 0:
        return ACTION_DOWN
    elif dy < 0:
        return ACTION_UP
    return ACTION_STAY


class GreedyAgent(AgentInterface):
    """
    Autonomous decentralized heuristic agent.
    - If carrying victim: navigates to closest shelter and unloads.
    - If idle: navigates to closest active victim and picks up.
    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id

    def act(self, agent_info: Dict[str, Any], obs: Dict[str, Any]) -> int:
        my_pos = tuple(agent_info["pos"])
        is_carrying = (
            agent_info.get("carrying") is not None
            or agent_info.get("state") == STATE_CARRYING
        )

        # 1. Carrying mode -> Head towards nearest shelter
        if is_carrying:
            shelters = obs.get("shelters", [])
            if not shelters:
                return ACTION_STAY

            # Find closest shelter
            nearest_shelter = min(
                shelters,
                key=lambda s: manhattan_distance(my_pos, tuple(s["pos"] if isinstance(s, dict) else s)),
            )
            shelter_pos = tuple(nearest_shelter["pos"] if isinstance(nearest_shelter, dict) else nearest_shelter)

            if my_pos == shelter_pos:
                return ACTION_DROPOFF
            return step_towards(my_pos, shelter_pos)

        # 2. Idle mode -> Head towards nearest active victim
        carried_ids = {
            victim_id
            for other in obs.get("agents", [])
            for victim_id in (other.get("inventory") or ([other.get("carrying")] if other.get("carrying") is not None else []))
        }
        active_victims = [
            v for v in obs.get("victims", [])
            if not v.get("rescued", False) and not v.get("died", False)
            and v.get("id") not in carried_ids
        ]

        if not active_victims:
            return ACTION_STAY

        # Find nearest victim (tie-breaker: highest urgency)
        target_victim = min(
            active_victims,
            key=lambda v: (
                manhattan_distance(my_pos, tuple(v["pos"])),
                -float(v.get("urgency", 0.0)),
            ),
        )
        target_pos = tuple(target_victim["pos"])

        if my_pos == target_pos:
            return ACTION_PICKUP

        return step_towards(my_pos, target_pos)


class MultiAgentController:
    """Wraps a list of AgentInterface agents into a unified joint controller."""

    def __init__(self, agents: List[AgentInterface]):
        self.agents = agents

    def act(self, obs: Dict[str, Any]) -> List[int]:
        actions = []
        agent_infos = obs.get("agents", [])
        for i, agent in enumerate(self.agents):
            agent_info = agent_infos[i] if i < len(agent_infos) else {"pos": (0, 0), "carrying": None}
            actions.append(agent.act(agent_info, obs))
        return actions
