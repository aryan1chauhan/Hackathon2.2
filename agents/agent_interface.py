"""
Abstract Agent Interface for Disaster Relief Tactical Units.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class AgentInterface(ABC):
    """Abstract base class for all agent controllers."""

    @abstractmethod
    def act(self, agent_info: Dict[str, Any], obs: Dict[str, Any]) -> int:
        """
        Calculates a discrete action [0..6] given local agent info and global observation.
        """
        raise NotImplementedError
