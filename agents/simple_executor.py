"""
Simple Executor Module

This module provides controllers for managing multiple agents and executing high-level task assignments.
It includes a MultiAgentController for independent agents and a TaskExecutor for coordinated task execution.
"""

from typing import List, Sequence, Optional, Dict, Any
from agents.agent_interface import AgentInterface


class MultiAgentController:
    """
    Controller that manages multiple independent AgentInterface instances.

    This controller holds a list of agents and delegates action selection to each agent
    based on their individual observations and states.

    Attributes:
        agents (List[AgentInterface]): List of agent instances.
    """

    def __init__(self, agents: Sequence[AgentInterface]):
        """
        Initialize the controller with a sequence of agents.

        Args:
            agents (Sequence[AgentInterface]): Sequence of agent instances to control.
        """
        self.agents = list(agents)

    def act(self, observation: Dict[str, Any]) -> List[int]:
        """
        Get actions from all agents for the current observation.

        Args:
            observation (Dict[str, Any]): Global observation of the environment.

        Returns:
            List[int]: List of actions, one for each agent.
        """
        actions = []
        # observation['agents'] provides agent info in order; map by id
        agents_info = {a['id']: a for a in observation['agents']}
        for ag in self.agents:
            info = agents_info.get(ag.agent_id)
            if info is None:
                # fallback: idle action
                actions.append(0)
            else:
                actions.append(ag.act(info, observation))
        return actions


class TaskExecutor:
    """
    Executor that follows high-level assignment strings for coordinated task execution.

    Assignments specify tasks like rescuing specific victims. The executor translates
    these high-level tasks into sequences of low-level actions.

    Supported assignment formats:
    - None or '': No operation (STAY)
    - 'rescue V{victim_id}': Go to victim position, PICKUP, then go to nearest shelter, DROPOFF

    Attributes:
        num_agents (int): Number of agents being controlled.
        assignments (Dict[int, Optional[str]]): Mapping of agent_id to assignment string.
        _targets (Dict[int, Optional[int]]): Internal mapping of agent_id to target victim ID.
    """

    def __init__(self, num_agents: int):
        """
        Initialize the TaskExecutor with a specified number of agents.

        Args:
            num_agents (int): Number of agents to control.
        """
        self.num_agents = num_agents
        # mapping agent_id -> assignment string
        self.assignments = {i: None for i in range(num_agents)}
        # internal state: agent_id -> target victim id
        self._targets = {i: None for i in range(num_agents)}

    def set_assignment(self, agent_id: int, assignment: Optional[str]):
        """
        Set an assignment for a specific agent.

        Args:
            agent_id (int): ID of the agent to assign the task to.
            assignment (Optional[str]): Assignment string or None.
        """
        self.assignments[agent_id] = assignment
        # parse assignment
        if assignment and assignment.strip().lower().startswith('rescue'):
            parts = assignment.strip().split()
            if len(parts) >= 2 and parts[1].startswith('V'):
                try:
                    vid = int(parts[1][1:])
                    self._targets[agent_id] = vid
                except Exception:
                    self._targets[agent_id] = None
        else:
            self._targets[agent_id] = None

    def act(self, observation: Dict[str, Any]) -> List[int]:
        """
        Generate actions for all agents based on their current assignments and state.

        Args:
            observation (Dict[str, Any]): Global observation of the environment.

        Returns:
            List[int]: List of actions, one for each agent.
        """
        from env.constants import ACTION_STAY, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_PICKUP, ACTION_DROPOFF

        actions = []
        agents_info = {a['id']: a for a in observation['agents']}
        victims = {v['id']: tuple(v['pos']) for v in observation.get('victims', [])}
        shelters = observation.get('shelters', [])

        for aid in range(self.num_agents):
            info = agents_info.get(aid)
            if info is None:
                actions.append(ACTION_STAY)
                continue

            apos = tuple(info['pos'])
            target_vid = self._targets.get(aid)

            # no target -> stay
            if target_vid is None or target_vid not in victims:
                actions.append(ACTION_STAY)
                continue

            vpos = victims[target_vid]

            # If agent not carrying and on victim -> pickup
            if info['carrying'] is None and apos == vpos:
                actions.append(ACTION_PICKUP)
                continue

            # If agent is carrying this victim, move to nearest shelter or dropoff if at shelter
            if info['carrying'] is not None and info['carrying'] == target_vid:
                if apos in shelters:
                    actions.append(ACTION_DROPOFF)
                    continue
                # move towards nearest shelter
                nearest = min(shelters, key=lambda s: abs(s[0] - apos[0]) + abs(s[1] - apos[1]))
                actions.append(self._move_towards(apos, tuple(nearest)))
                continue

            # Otherwise move towards victim
            actions.append(self._move_towards(apos, vpos))

        return actions

    @staticmethod
    def _move_towards(src: tuple, dst: tuple) -> int:
        """
        Calculate the action to move from source towards destination.

        Args:
            src (tuple): Source position (x, y).
            dst (tuple): Destination position (x, y).

        Returns:
            int: Action ID for movement.
        """
        sx, sy = src
        dx, dy = dst
        from env.constants import ACTION_STAY, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT
        if sx < dx:
            return ACTION_RIGHT
        if sx > dx:
            return ACTION_LEFT
        if sy < dy:
            return ACTION_DOWN
        if sy > dy:
            return ACTION_UP
        return ACTION_STAY
