"""
Greedy Agent Module

This module implements a greedy agent that makes decisions based on simple heuristics:
- If carrying a victim, move to the nearest shelter and drop off.
- If not carrying, move to the nearest victim and pick up.
"""

from typing import Dict, Any, Tuple
from env.constants import ACTION_STAY, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_PICKUP, ACTION_DROPOFF
from agents.agent_interface import AgentInterface


def _move_towards(src: Tuple[int, int], dst: Tuple[int, int]) -> int:
    """
    Calculate the action to move from source position towards destination.

    Args:
        src (Tuple[int, int]): Source position (x, y).
        dst (Tuple[int, int]): Destination position (x, y).

    Returns:
        int: Action ID for movement (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, or ACTION_STAY).
    """
    sx, sy = src
    dx, dy = dst
    if sx < dx:
        return ACTION_RIGHT
    if sx > dx:
        return ACTION_LEFT
    if sy < dy:
        return ACTION_DOWN
    if sy > dy:
        return ACTION_UP
    return ACTION_STAY


class GreedyAgent(AgentInterface):
    """
    A greedy agent that prioritizes immediate local optima.

    The agent follows a simple strategy:
    - If carrying a victim, navigate to the nearest shelter and drop off.
    - If not carrying, navigate to the nearest victim and pick up.

    This approach is computationally efficient but may not always lead to globally optimal solutions.
    """

    def act(self, agent_info: Dict[str, Any], observation: Dict[str, Any]) -> int:
        """
        Determine the next action based on greedy heuristics.

        Args:
            agent_info (Dict[str, Any]): Information about the agent's current state.
            observation (Dict[str, Any]): Global observation of the environment.

        Returns:
            int: Action ID to perform.
        """
        apos = tuple(agent_info['pos'])
        carrying = agent_info['carrying']
        shelters = observation['shelters']
        victims = observation['victims']

        if carrying is not None:
            # If at shelter, drop off
            if apos in shelters:
                return ACTION_DROPOFF
            # Move to nearest shelter
            nearest = min(shelters, key=lambda s: abs(s[0] - apos[0]) + abs(s[1] - apos[1]))
            return _move_towards(apos, tuple(nearest))

        # Not carrying
        if not victims:
            return ACTION_STAY

        victims_pos = [tuple(v['pos']) for v in victims]
        nearest_v = min(victims_pos, key=lambda p: abs(p[0] - apos[0]) + abs(p[1] - apos[1]))
        if apos == nearest_v:
            return ACTION_PICKUP
        return _move_towards(apos, nearest_v)
