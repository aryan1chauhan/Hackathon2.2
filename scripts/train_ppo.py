"""
PPO Policy Training Pipeline for Multi-Agent Disaster Relief using Stable-Baselines3.
Trains a neural policy mapping spatial vector observations to joint agent action vectors.
"""

import os
import sys
import argparse

# Ensure local module priority
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.rl_utils import DisasterGymWrapper


def train_ppo(timesteps: int = 2000, model_save_path: str = "models/ppo_agent.zip"):
    print("=" * 65)
    print("AEGISCOORD // PPO REINFORCEMENT LEARNING BASELINE")
    print(f"Target Timesteps: {timesteps} | Save Path: {model_save_path}")
    print("=" * 65)

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("[ERROR] stable_baselines3 is not installed. Run: pip install stable-baselines3")
        return

    env = DisasterGymWrapper(grid_size=10, num_agents=3, max_victims=6, seed=42)

    # Multi-Input / MLP policy for discrete multi-agent actions
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        verbose=1,
    )

    print("\nStarting PPO optimization...")
    model.learn(total_timesteps=timesteps)

    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    model.save(model_save_path)
    print(f"\n[SUCCESS] Model successfully trained and saved to {model_save_path}")

    # Evaluate 1 test episode
    obs, info = env.reset(seed=123)
    total_reward = 0.0
    for step in range(30):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"Evaluation Test Episode Finished | Total Reward: {total_reward:.2f} | Rescued: {env.env.total_rescued} | Casualties: {env.env.total_died}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training Pipeline")
    parser.add_argument("--timesteps", type=int, default=1500)
    parser.add_argument("--save-path", type=str, default="models/ppo_agent.zip")
    args = parser.parse_args()

    train_ppo(timesteps=args.timesteps, model_save_path=args.save_path)
