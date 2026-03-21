"""
Task 1: Training Script for DQN Agent on Atari Environment
This script trains a DQN agent using Stable Baselines3 and Gymnasium.
It includes hyperparameter tuning with 10 different configurations.
"""

import os
import re
import gymnasium as gym
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.policies import MlpPolicy, CnnPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
import warnings

warnings.filterwarnings('ignore')

try:
    import ale_py
except ImportError:
    ale_py = None


class TrainingLogger(BaseCallback):
    """Custom callback to log training details."""
    
    def __init__(self, log_file='training_log.txt'):
        super(TrainingLogger, self).__init__()
        self.log_file = log_file
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Log data every 1000 steps
        if self.num_timesteps % 1000 == 0:
            ep_info = getattr(self.model, 'ep_info_buffer', None)
            if ep_info:
                rewards = [ep.get('r', 0.0) for ep in ep_info if isinstance(ep, dict)]
                lengths = [ep.get('l', 0) for ep in ep_info if isinstance(ep, dict)]
                mean_reward = float(np.mean(rewards)) if rewards else 0.0
                mean_length = float(np.mean(lengths)) if lengths else 0.0
                with open(self.log_file, 'a') as f:
                    f.write(
                        f"Timestep: {self.num_timesteps}, "
                        f"Mean Reward (last 100): {mean_reward:.2f}, "
                        f"Mean Episode Length (last 100): {mean_length:.2f}\n"
                    )
        return True


def create_environment(env_name='ALE/Pong-v5'):
    """
    Create and wrap the Atari environment.
    
    Args:
        env_name (str): Name of the Atari environment
        
    Returns:
        gym.Env: The created environment
    """
    if ale_py is None:
        raise ImportError(
            "ale-py is not installed. Install dependencies with: pip install -r requirements.txt"
        )

    # Gymnasium requires explicit registration for ALE environments.
    gym.register_envs(ale_py)
    env = gym.make(env_name)
    # Apply Atari preprocessing to reduce observation size and stabilize learning.
    env = AtariWrapper(env)
    return env


def train_dqn_agent(
    env_name='ALE/Pong-v5',
    policy_type='CnnPolicy',
    total_timesteps=100000,
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=32,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    buffer_size=100000,
    model_save_path='dqn_model.zip',
    log_file='training_log.txt',
    experiment_name="Default"
):
    """
    Train a DQN agent on the specified Atari environment.
    
    Args:
        env_name (str): Name of the Atari environment
        policy_type (str): Type of policy ('MlpPolicy' or 'CnnPolicy')
        total_timesteps (int): Total timesteps to train
        learning_rate (float): Learning rate for the optimizer
        gamma (float): Discount factor
        batch_size (int): Batch size for training
        epsilon_start (float): Initial exploration rate
        epsilon_end (float): Final exploration rate
        epsilon_decay (float): Decay rate for epsilon
        buffer_size (int): Replay buffer size
        model_save_path (str): Path to save the trained model
        log_file (str): Path to save training logs
        experiment_name (str): Name of the experiment for documentation
        
    Returns:
        tuple: (trained_model, environment, training_metrics)
    """
    
    print(f"\n{'='*70}")
    print(f"Starting Training: {experiment_name}")
    print(f"{'='*70}")
    print(f"Environment: {env_name}")
    print(f"Policy Type: {policy_type}")
    print(f"Hyperparameters:")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Gamma: {gamma}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Epsilon: start={epsilon_start}, end={epsilon_end}, decay={epsilon_decay}")
    print(f"  - Replay Buffer Size: {buffer_size}")
    print(f"  - Total Timesteps: {total_timesteps}")
    print(f"{'='*70}\n")

    # Ensure output directories exist for nested paths.
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.dirname(model_save_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    # Create environment
    env = create_environment(env_name)
    
    # Select policy
    policy = CnnPolicy if policy_type == 'CnnPolicy' else MlpPolicy
    
    # Clear or create log file
    with open(log_file, 'w') as f:
        f.write(f"Training Log - {experiment_name}\n")
        f.write(f"Started at: {datetime.now()}\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Policy: {policy_type}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Gamma: {gamma}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Epsilon: start={epsilon_start}, end={epsilon_end}, decay={epsilon_decay}\n")
        f.write(f"Replay Buffer Size: {buffer_size}\n")
        f.write("-" * 70 + "\n\n")
    
    # Create the DQN model
    model = DQN(
        policy,
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        exploration_initial_eps=epsilon_start,
        exploration_final_eps=epsilon_end,
        exploration_fraction=epsilon_decay,
        verbose=1,
        device='auto'
    )
    
    # Train the model
    print("Training in progress...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=TrainingLogger(log_file)
    )
    
    # Save the trained model
    os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.', exist_ok=True)
    model.save(model_save_path)
    print(f"\nModel saved to: {model_save_path}\n")
    
    # Log completion
    with open(log_file, 'a') as f:
        f.write(f"\nTraining completed at: {datetime.now()}\n")
        f.write(f"Model saved to: {model_save_path}\n")
    
    return model, env, {
        'environment': env_name,
        'policy': policy_type,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'batch_size': batch_size,
        'epsilon_start': epsilon_start,
        'epsilon_end': epsilon_end,
        'epsilon_decay': epsilon_decay,
        'buffer_size': buffer_size,
        'total_timesteps': total_timesteps,
        'model_path': model_save_path
    }


def run_hyperparameter_experiments(member_names=None):
    """
    Run 10 different hyperparameter combinations (required by rubric).
    Each group member must experiment with 10 different configurations.
    """

    def summarize_noted_behavior(log_file_path):
        """Build a short observed-behavior summary from training logs."""
        if not os.path.exists(log_file_path):
            return "No log data found"

        reward_pattern = re.compile(r"Mean Reward \(last 100\):\s*(-?\d+(?:\.\d+)?)")
        length_pattern = re.compile(r"Mean Episode Length \(last 100\):\s*(-?\d+(?:\.\d+)?)")
        rewards = []
        lengths = []

        with open(log_file_path, 'r') as f:
            for line in f:
                reward_match = reward_pattern.search(line)
                if reward_match:
                    rewards.append(float(reward_match.group(1)))
                length_match = length_pattern.search(line)
                if length_match:
                    lengths.append(float(length_match.group(1)))

        if not rewards:
            return "Insufficient reward samples"

        window = min(3, len(rewards))
        start_reward = float(np.mean(rewards[:window]))
        end_reward = float(np.mean(rewards[-window:]))
        reward_delta = end_reward - start_reward

        if reward_delta > 1.0:
            reward_trend = "Reward improved"
        elif reward_delta < -1.0:
            reward_trend = "Reward declined"
        else:
            reward_trend = "Reward mostly stable"

        stability = float(np.std(rewards)) if len(rewards) > 1 else 0.0
        stability_note = "stable" if stability < 3.0 else "high variance"

        if lengths:
            start_length = float(np.mean(lengths[:window]))
            end_length = float(np.mean(lengths[-window:]))
            length_delta = end_length - start_length
            if length_delta > 1.0:
                length_note = "episode length increased"
            elif length_delta < -1.0:
                length_note = "episode length decreased"
            else:
                length_note = "episode length steady"
        else:
            length_note = "episode length unavailable"

        return (
            f"{reward_trend} ({start_reward:.2f}->{end_reward:.2f}), "
            f"{stability_note}, {length_note}"
        )
    
    # Create output folders for experiment artifacts.
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    hyperparameter_configs = [
    {
        'name': 'Stecie_Exp1_LowLR_LowGamma',
        'learning_rate': 1e-5,
        'gamma': 0.90,
        'batch_size': 32,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.995
    },
    {
        'name': 'Stecie_Exp2_HighLR_VeryHighGamma',
        'learning_rate': 5e-4,
        'gamma': 0.999,
        'batch_size': 32,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.995
    },
    {
        'name': 'Stecie_Exp3_UltraLargeBatch',
        'learning_rate': 2e-4,
        'gamma': 0.99,
        'batch_size': 256,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.995
    },
    {
        'name': 'Stecie_Exp4_UltraSmallBatch',
        'learning_rate': 2e-4,
        'gamma': 0.99,
        'batch_size': 4,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.995
    },
    {
        'name': 'Stecie_Exp5_NoDecay_Epsilon',
        'learning_rate': 2e-4,
        'gamma': 0.99,
        'batch_size': 32,
        'epsilon_start': 1.0,
        'epsilon_end': 0.5,
        'epsilon_decay': 1.0
    },
    {
        'name': 'Stecie_Exp6_FastDecay_Short',
        'learning_rate': 2e-4,
        'gamma': 0.99,
        'batch_size': 32,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.95
    },
    {
        'name': 'Stecie_Exp7_HighLR_LowGamma_SmallBatch',
        'learning_rate': 8e-4,
        'gamma': 0.92,
        'batch_size': 8,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.995
    },
    {
        'name': 'Stecie_Exp8_LowLR_HighGamma_LargeBatch',
        'learning_rate': 1e-5,
        'gamma': 0.995,
        'batch_size': 128,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.995
    },
    {
        'name': 'Stecie_Exp9_ExtremeExploration',
        'learning_rate': 2e-4,
        'gamma': 0.99,
        'batch_size': 32,
        'epsilon_start': 1.0,
        'epsilon_end': 0.2,
        'epsilon_decay': 0.98
    },
    {
        'name': 'Stecie_Exp10_MidRange_All',
        'learning_rate': 3e-4,
        'gamma': 0.97,
        'batch_size': 64,
        'epsilon_start': 0.8,
        'epsilon_end': 0.03,
        'epsilon_decay': 0.99
    }
]
    
    if member_names is None:
        member_names = ["Yassin"]
    member_names = [name.strip() for name in member_names if name and name.strip()]
    if not member_names:
        member_names = ["Yassin"]

    # Store results for documentation
    results = []
    
    # Run each experiment
    for idx, config in enumerate(hyperparameter_configs):
        print(f"\n\nRunning {config['name']}...")
        assigned_member = member_names[(idx // 10) % len(member_names)]
        experiment_log_path = f"logs/{config['name']}_training.txt"
        
        model, env, metrics = train_dqn_agent(
            env_name='ALE/Pong-v5',
            policy_type='CnnPolicy',
            total_timesteps=50000,  # Reduced for faster training
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            batch_size=config['batch_size'],
            epsilon_start=config['epsilon_start'],
            epsilon_end=config['epsilon_end'],
            epsilon_decay=config['epsilon_decay'],
            model_save_path=f"models/{config['name']}_dqn_model.zip",
            log_file=experiment_log_path,
            experiment_name=config['name']
        )

        noted_behavior = summarize_noted_behavior(experiment_log_path)
        
        # Save results
        results.append({
            'Member Name': assigned_member,
            'Experiment': config['name'],
            'Learning Rate': config['learning_rate'],
            'Gamma': config['gamma'],
            'Batch Size': config['batch_size'],
            'Epsilon Start': config['epsilon_start'],
            'Epsilon End': config['epsilon_end'],
            'Epsilon Decay': config['epsilon_decay'],
            'Noted Behavior': noted_behavior,
            'Model Path': f"models/{config['name']}_dqn_model.zip"
        })
        
        env.close()
    
    # Create and save results table
    df_results = pd.DataFrame(results)

    csv_path = 'hyperparameter_experiments.csv'
    md_path = 'HYPERPARAMETER_TUNING_TABLE.md'

    # Keep updating aggregate results across runs.
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        df_results = pd.concat([existing_df, df_results], ignore_index=True)
        df_results = df_results.drop_duplicates(subset=['Member Name', 'Experiment'], keep='last')
    
    # Save as CSV
    df_results.to_csv(csv_path, index=False, encoding='utf-8')
    
    # Save as formatted table for documentation
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Hyperparameter Tuning Experiments - Task 1\n\n")
        f.write("## Summary Table\n\n")
        try:
            f.write(df_results.to_markdown(index=False))
        except ImportError:
            # Fallback when optional markdown dependency (tabulate) is unavailable.
            f.write(df_results.to_string(index=False))
        f.write("\n\n")
        f.write("## Detailed Results\n\n")
        for idx, row in df_results.iterrows():
            f.write(f"### Experiment {idx + 1}: {row['Experiment']}\n")
            f.write(f"- **Member Name**: {row['Member Name']}\n")
            f.write(f"- **Learning Rate (lr)**: {row['Learning Rate']}\n")
            f.write(f"- **Gamma (γ)**: {row['Gamma']}\n")
            f.write(f"- **Batch Size**: {row['Batch Size']}\n")
            f.write(f"- **Epsilon Start**: {row['Epsilon Start']}\n")
            f.write(f"- **Epsilon End**: {row['Epsilon End']}\n")
            f.write(f"- **Epsilon Decay**: {row['Epsilon Decay']}\n")
            f.write(f"- **Noted Behavior**: {row['Noted Behavior']}\n")
            f.write(f"- **Model Path**: {row['Model Path']}\n\n")
    
    print("\n" + "="*70)
    print("All 10 Experiments Completed!")
    print("="*70)
    print("\nResults saved to:")
    print("  - hyperparameter_experiments.csv")
    print("  - HYPERPARAMETER_TUNING_TABLE.md")
    print("\nModels saved to:")
    print("  - models/ directory")
    print("\nTraining logs saved to:")
    print("  - logs/ directory")
    
    return df_results


def train_final_model(policy_type='CnnPolicy'):
    """
    Train the final best model for use in play.py.
    Uses a balanced, well-tested configuration.
    """
    print("\n" + "="*70)
    print("Training Final Model for Deployment")
    print("="*70)
    
    model, env, metrics = train_dqn_agent(
        env_name='ALE/Pong-v5',
        policy_type=policy_type,
        total_timesteps=100000,
        learning_rate=2e-4,
        gamma=0.99,
        batch_size=32,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        model_save_path='dqn_model.zip',
        log_file='final_training.txt',
        experiment_name='Final_Model_Deployment'
    )
    
    env.close()
    return model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN Agent on Atari')
    parser.add_argument('--mode', type=str, default='final', 
                       choices=['final', 'experiments'],
                       help='Training mode: final (train single model) or experiments (run all 10 configs)')
    parser.add_argument('--policy', type=str, default='CnnPolicy',
                       choices=['CnnPolicy', 'MlpPolicy'],
                       help='Policy type to use')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total timesteps for training')
    parser.add_argument('--env', type=str, default='ALE/Pong-v5',
                       help='Atari environment name')
    parser.add_argument('--members', type=str, default='Yassin',
                       help='Comma-separated member names, e.g. Yassin,Alice,Bob')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    if args.mode == 'final':
        # Train single final model
        print("Training mode: FINAL MODEL")
        train_final_model(policy_type=args.policy)
    else:
        # Run all 10 hyperparameter experiments
        print("Training mode: HYPERPARAMETER EXPERIMENTS (10 Configurations)")
        member_names = [name.strip() for name in args.members.split(',') if name.strip()]
        results_df = run_hyperparameter_experiments(member_names=member_names)
        print("\n" + results_df.to_string())
    
    print("\nTraining completed successfully!")
