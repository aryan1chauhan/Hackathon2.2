"""
Agent Interface Module

This module defines the base interface for all agent implementations in the disaster relief simulation.
All agents must inherit from AgentInterface and implement the act method.
"""

from typing import Dict, Any


class AgentInterface:
    """
    Base class for all agent implementations.

    Agents are responsible for making decisions based on their current state and the global observation.
    Each agent has a unique ID and must implement the act method to return an action.

    Attributes:
        agent_id (int): Unique identifier for the agent.
    """

    def __init__(self, agent_id: int):
        """
        Initialize the agent with a unique ID.

        Args:
            agent_id (int): Unique identifier for the agent.
        """
        self.agent_id = agent_id

    def act(self, agent_info: Dict[str, Any], observation: Dict[str, Any]) -> int:
        """
        Determine the next action for the agent.

        Args:
            agent_info (Dict[str, Any]): Information about the agent's current state,
                including position, carrying status, etc.
            observation (Dict[str, Any]): Global observation of the environment,
                including all agents, victims, shelters, etc.

        Returns:
            int: Action ID to perform (0-6 corresponding to stay, move directions, pickup, dropoff).

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()
