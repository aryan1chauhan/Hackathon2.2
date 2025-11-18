#!/usr/bin/env python3
"""
Training script for PPO agents in the disaster relief environment.
Trains separate models for each agent or a shared model.
"""

import os
import sys
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# Add parent directory to path to import agents module
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agents.rl_utils import DisasterReliefGymWrapper


def train_ppo_agent(agent_id: int = 0, total_timesteps: int = 100000, model_dir: str = "models"):
    """
    Train a PPO model for a specific agent.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Create environment
    env = DisasterReliefGymWrapper()
    eval_env = DisasterReliefGymWrapper()

    # PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"./ppo_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best_ppo_agent_{agent_id}/",
        log_path=f"./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # Train
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save final model
    model.save(f"{model_dir}/ppo_agent_{agent_id}.zip")
    print(f"Model saved to {model_dir}/ppo_agent_{agent_id}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for disaster relief")
    parser.add_argument("--agent_id", type=int, default=0, help="Agent ID to train")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")

    args = parser.parse_args()
    train_ppo_agent(args.agent_id, args.timesteps, args.model_dir)
