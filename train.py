"""
Task 1: Training Script for DQN Agent on Atari Environment
This script trains a DQN agent using Stable Baselines3 and Gymnasium.
It includes hyperparameter tuning with 10 different configurations.
"""

import os
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


def run_hyperparameter_experiments():
    """
    Run 10 different hyperparameter combinations (required by rubric).
    Each group member must experiment with 10 different configurations.
    """
    
    # Create output folders for experiment artifacts.
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Define 10 different hyperparameter configurations
    hyperparameter_configs = [
        # Yassin hyperparameter configurations
        {
            'name': 'Exp1_HighLR_LowGamma',
            'learning_rate': 1e-3,
            'gamma': 0.95,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.995
        },
        {
            'name': 'Exp2_LowLR_HighGamma',
            'learning_rate': 5e-5,
            'gamma': 0.99,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.995
        },
        {
            'name': 'Exp3_LargeBatch_HighEpsilon',
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'batch_size': 64,
            'epsilon_start': 1.0,
            'epsilon_end': 0.1,
            'epsilon_decay': 0.99
        },
        {
            'name': 'Exp4_SmallBatch_LowEpsilon',
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'batch_size': 16,
            'epsilon_start': 0.5,
            'epsilon_end': 0.02,
            'epsilon_decay': 0.98
        },
        {
            'name': 'Exp5_MediumLR_MediumGamma',
            'learning_rate': 3e-4,
            'gamma': 0.97,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.995
        },
        {
            'name': 'Exp6_HighLR_HighGamma_LargeBatch',
            'learning_rate': 5e-4,
            'gamma': 0.995,
            'batch_size': 64,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.99
        },
        {
            'name': 'Exp7_LowLR_LowGamma_SmallBatch',
            'learning_rate': 1e-5,
            'gamma': 0.9,
            'batch_size': 16,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.99
        },
        {
            'name': 'Exp8_BalancedConfig',
            'learning_rate': 2e-4,
            'gamma': 0.99,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.995
        },
        {
            'name': 'Exp9_AggressiveExploration',
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.1,
            'epsilon_decay': 0.985
        },
        {
            'name': 'Exp10_ConservativeConfig',
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'batch_size': 32,
            'epsilon_start': 0.9,
            'epsilon_end': 0.02,
            'epsilon_decay': 0.998
        }
    ]
    
    # Store results for documentation
    results = []
    
    # Run each experiment
    for config in hyperparameter_configs:
        print(f"\n\nRunning {config['name']}...")
        
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
            log_file=f"logs/{config['name']}_training.txt",
            experiment_name=config['name']
        )
        
        # Save results
        results.append({
            'Experiment': config['name'],
            'Learning Rate': config['learning_rate'],
            'Gamma': config['gamma'],
            'Batch Size': config['batch_size'],
            'Epsilon Start': config['epsilon_start'],
            'Epsilon End': config['epsilon_end'],
            'Epsilon Decay': config['epsilon_decay'],
            'Model Path': f"models/{config['name']}_dqn_model.zip"
        })
        
        env.close()
    
    # Create and save results table
    df_results = pd.DataFrame(results)
    
    # Save as CSV
    df_results.to_csv('hyperparameter_experiments.csv', index=False)
    
    # Save as formatted table for documentation
    with open('HYPERPARAMETER_TUNING_TABLE.md', 'w') as f:
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
            f.write(f"- **Learning Rate (lr)**: {row['Learning Rate']}\n")
            f.write(f"- **Gamma (γ)**: {row['Gamma']}\n")
            f.write(f"- **Batch Size**: {row['Batch Size']}\n")
            f.write(f"- **Epsilon Start**: {row['Epsilon Start']}\n")
            f.write(f"- **Epsilon End**: {row['Epsilon End']}\n")
            f.write(f"- **Epsilon Decay**: {row['Epsilon Decay']}\n")
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
        results_df = run_hyperparameter_experiments()
        print("\n" + results_df.to_string())
    
    print("\nTraining completed successfully!")
