import os
import sys
import numpy as np

# Ensure local module priority
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.rl_utils import DisasterGymWrapper
from agents.ppo_agent import PPOFleetPolicy
from env.constants import ACTION_STAY


def test_gym_wrapper_shapes_and_reset():
    """Verify Gymnasium wrapper observation and action space dimensions."""
    env = DisasterGymWrapper(grid_size=10, num_agents=3, max_victims=6, seed=42)
    obs, info = env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == env.observation_space.shape
    assert len(env.action_space.nvec) == 3
    assert all(n == 7 for n in env.action_space.nvec)


def test_gym_wrapper_step_transition():
    """Verify Gymnasium environment stepping and reward computation."""
    env = DisasterGymWrapper(grid_size=10, num_agents=3, max_victims=6, seed=42)
    obs, info = env.reset(seed=42)

    # Step with valid actions
    next_obs, reward, terminated, truncated, step_info = env.step([0, 1, 2])
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "step" in step_info


def test_ppo_fleet_policy_fallback_and_prediction():
    """Verify PPO fleet policy loader gracefully returns valid actions."""
    policy = PPOFleetPolicy("non_existent_model.zip")
    dummy_obs = np.zeros(52, dtype=np.float32)
    actions = policy.act_joint(dummy_obs)
    assert len(actions) == 3
    assert all(a == ACTION_STAY for a in actions)
