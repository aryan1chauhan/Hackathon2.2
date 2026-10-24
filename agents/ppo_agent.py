"""
Autonomous Neural Policy Agent utilizing trained Stable-Baselines3 PPO models.
"""

import os
import sys
from typing import Dict, Any, Optional, List
import numpy as np

# Ensure local module priority
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent_interface import AgentInterface
from env.constants import ACTION_STAY


class PPOFleetPolicy:
    """Joint fleet controller powered by a trained Stable-Baselines3 PPO checkpoint."""

    def __init__(self, model_path: str = "models/ppo_agent.zip"):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.isfile(self.model_path):
            try:
                from stable_baselines3 import PPO
                self.model = PPO.load(self.model_path)
            except Exception:
                self.model = None

    def act_joint(self, flat_obs: np.ndarray, deterministic: bool = True) -> List[int]:
        if self.model is not None:
            actions, _ = self.model.predict(flat_obs, deterministic=deterministic)
            return [int(a) for a in actions]
        return [ACTION_STAY, ACTION_STAY, ACTION_STAY]
