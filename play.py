"""
Task 2: Playing Script
Loads the best trained DQN model and runs it in the Atari Pong environment.

The agent uses a greedy policy during evaluation — it always selects the action
with the highest Q-value, with no random exploration (epsilon = 0).

Usage:
    python play.py

Optional arguments:
    --model     Path to the trained model (default: dqn_model.zip)
    --episodes  Number of episodes to play (default: 5)
    --no-render Disable game rendering (useful for headless evaluation)
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
import warnings

warnings.filterwarnings("ignore")

try:
    import ale_py
except ImportError:
    ale_py = None


# ---------------------------------------------------------------------------
# Environment factory — matches the training environment exactly
# ---------------------------------------------------------------------------
def create_environment(env_name="ALE/Pong-v5", render=True):
    """
    Create the Atari Pong environment for evaluation.
    Uses render_mode='human' to display the game on screen.

    Args:
        env_name (str): Name of the Atari environment
        render (bool): Whether to render the game visually

    Returns:
        gym.Env: The wrapped environment
    """
    if ale_py is None:
        raise ImportError("ale-py not installed. Run: pip install -r requirements.txt")

    gym.register_envs(ale_py)
    render_mode = "human" if render else "rgb_array"
    env = gym.make(env_name, render_mode=render_mode)
    env = AtariWrapper(env)
    return env


# ---------------------------------------------------------------------------
# Greedy policy evaluation
# ---------------------------------------------------------------------------
def evaluate_agent(model, env, num_episodes=5):
    """
    Run the agent for a given number of episodes using a greedy policy.
    The agent always picks the action with the highest Q-value — no exploration.

    Args:
        model: The loaded DQN model
        env: The Atari environment
        num_episodes (int): Number of episodes to run

    Returns:
        list: Total rewards per episode
    """
    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step = 0

        print(f"\nEpisode {episode} starting...")

        while not done:
            # Greedy action selection — deterministic=True disables exploration
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1

        episode_rewards.append(total_reward)
        print(f"Episode {episode} finished — Total Reward: {total_reward:.1f}  Steps: {step}")

    return episode_rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Play Atari Pong with a trained DQN agent")
    parser.add_argument(
        "--model",
        type=str,
        default="dqn_model.zip",
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to play",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable game rendering",
    )
    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        print("Make sure the model path is correct and the file exists.")
        return

    print("=" * 70)
    print("Task 2: DQN Agent Evaluation — Atari Pong")
    print("=" * 70)
    print(f"Model : {args.model}")
    print(f"Episodes : {args.episodes}")
    print(f"Policy : Greedy (deterministic=True, no exploration)")
    print(f"Rendering : {'Disabled' if args.no_render else 'Enabled'}")
    print("=" * 70)

    # Load the trained model
    print("\nLoading model...")
    model = DQN.load(args.model)
    print("Model loaded successfully.")

    # Create environment
    print("Setting up environment...")
    env = create_environment(render=not args.no_render)
    print("Environment ready.\n")

    # Run evaluation episodes
    rewards = evaluate_agent(model, env, num_episodes=args.episodes)

    # Summary
    print("\n" + "=" * 70)
    print("Evaluation Complete")
    print("=" * 70)
    print(f"Episodes played   : {args.episodes}")
    print(f"Mean reward       : {np.mean(rewards):.2f}")
    print(f"Best episode      : {max(rewards):.1f}")
    print(f"Worst episode     : {min(rewards):.1f}")
    print(f"All rewards       : {[round(r, 1) for r in rewards]}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
