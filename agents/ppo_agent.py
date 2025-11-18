"""
PPO Agent Module

This module implements a PPO (Proximal Policy Optimization) based agent that uses a trained RL model
for action selection. The agent loads a pre-trained model and uses it to make decisions.
"""

import os
from typing import Dict, Any, Optional
from stable_baselines3 import PPO
from agents.agent_interface import AgentInterface
from agents.rl_utils import flatten_observation


class PPOAgent(AgentInterface):
    """
    PPO-based agent that uses a trained RL model for action selection.

    This agent assumes a pre-trained PPO model is available and loads it for inference.
    Each agent can have its own model, allowing for specialized training per agent.

    Attributes:
        model_path (str): Path to the trained PPO model file.
        model (PPO): Loaded PPO model instance.
    """

    def __init__(self, agent_id: int, model_path: Optional[str] = None):
        """
        Initialize the PPO agent with a model path.

        Args:
            agent_id (int): Unique identifier for the agent.
            model_path (Optional[str]): Path to the trained model file. If None,
                defaults to "models/ppo_agent_{agent_id}.zip".

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        super().__init__(agent_id)
        self.model_path = model_path or f"models/ppo_agent_{agent_id}.zip"
        if os.path.exists(self.model_path):
            self.model = PPO.load(self.model_path)
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}. Train the model first.")

    def act(self, agent_info: Dict[str, Any], observation: Dict[str, Any]) -> int:
        """
        Select action using the PPO model.

        The full environment observation is flattened into a 1D array for the model input.
        The model predicts actions deterministically for consistent behavior.

        Args:
            agent_info (Dict[str, Any]): Information about the agent's current state.
            observation (Dict[str, Any]): Global observation of the environment.

        Returns:
            int: Action ID predicted by the model.

        Note:
            For multi-agent scenarios, this assumes each agent has its own trained model.
        """
        flat_obs = flatten_observation(observation)
        action, _ = self.model.predict(flat_obs, deterministic=True)
        return int(action)
