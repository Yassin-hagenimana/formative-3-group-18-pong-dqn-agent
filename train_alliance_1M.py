"""
Task 1: Extended Hyperparameter Experiments — 1 Million Timesteps
Trains 5 DQN configurations on ALE/Pong-v5 using Stable Baselines3 at 1M timesteps.

The purpose of these experiments is twofold:
  1. Observe clearer behavioral differences that only become visible with more training.
  2. Directly compare select configurations against their 50k counterparts to isolate
     the effect of increased timesteps on agent performance.

Run all 5 experiments:
    python train_alliance_1M.py --mode experiments

Run the best config (1M_Exp1_Balanced) as a standalone model:
    python train_alliance_1M.py --mode final
"""

import os
import re
import gymnasium as gym
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy, CnnPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
import warnings

warnings.filterwarnings("ignore")

try:
    import ale_py
except ImportError:
    ale_py = None


# ---------------------------------------------------------------------------
# Hyperparameter configurations — 5 experiments at 1M timesteps
#
# Each experiment is intentionally paired with a 50k counterpart to allow
# direct comparison of the timestep effect on the same configuration.
#
# 1M_Exp1 — Balanced baseline: standard config to show what 1M timesteps
#            alone achieves. Paired with Yassin's Exp8_BalancedConfig.
#
# 1M_Exp2 — Low gamma (0.80): same as Alliance 50k Exp2. Direct comparison
#            to reveal whether short-sightedness hurts more or less with
#            longer training.
#
# 1M_Exp3 — XL batch (128): same as Alliance 50k Exp3. Tests whether the
#            stability advantage of large batches becomes more visible at 1M.
#
# 1M_Exp4 — High learning rate: tests whether a high LR converges faster
#            or causes instability when given more timesteps to play out.
#
# 1M_Exp5 — Near-zero epsilon end (0.005): tests whether committing almost
#            fully to exploitation pays off when the agent has had enough
#            training to build reliable Q-values.
# ---------------------------------------------------------------------------
HYPERPARAMETER_CONFIGS_1M = [
    {
        "name": "1M_Exp1_Balanced",
        "learning_rate": 0.0001,
        "gamma": 0.99,
        "batch_size": 32,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.995,
    },
    {
        "name": "1M_Exp2_LowGamma",
        "learning_rate": 0.0001,
        "gamma": 0.80,
        "batch_size": 32,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.995,
    },
    {
        "name": "1M_Exp3_XLBatch",
        "learning_rate": 0.0001,
        "gamma": 0.99,
        "batch_size": 128,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.995,
    },
    {
        "name": "1M_Exp4_HighLR",
        "learning_rate": 0.001,
        "gamma": 0.99,
        "batch_size": 32,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.995,
    },
    {
        "name": "1M_Exp5_LowEpsEnd",
        "learning_rate": 0.0001,
        "gamma": 0.99,
        "batch_size": 32,
        "epsilon_start": 1.0,
        "epsilon_end": 0.005,
        "epsilon_decay": 0.995,
    },
]


# ---------------------------------------------------------------------------
# Training Logger Callback
# ---------------------------------------------------------------------------
class TrainingLogger(BaseCallback):
    """Logs mean reward and episode length every 1000 timesteps."""

    def __init__(self, log_file="training_log.txt"):
        super().__init__()
        self.log_file = log_file

    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:
            ep_info = getattr(self.model, "ep_info_buffer", None)
            if ep_info:
                rewards = [ep.get("r", 0.0) for ep in ep_info if isinstance(ep, dict)]
                lengths = [ep.get("l", 0) for ep in ep_info if isinstance(ep, dict)]
                mean_reward = float(np.mean(rewards)) if rewards else 0.0
                mean_length = float(np.mean(lengths)) if lengths else 0.0
                with open(self.log_file, "a") as f:
                    f.write(
                        f"Timestep: {self.num_timesteps}, "
                        f"Mean Reward (last 100): {mean_reward:.2f}, "
                        f"Mean Episode Length (last 100): {mean_length:.2f}\n"
                    )
        return True


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def create_environment(env_name="ALE/Pong-v5"):
    if ale_py is None:
        raise ImportError("ale-py not installed. Run: pip install -r requirements.txt")
    gym.register_envs(ale_py)
    env = gym.make(env_name)
    env = AtariWrapper(env)
    return env


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------
def train_dqn_agent(
    env_name="ALE/Pong-v5",
    policy_type="CnnPolicy",
    total_timesteps=1000000,
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=32,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    buffer_size=100000,
    model_save_path="dqn_model.zip",
    log_file="training_log.txt",
    experiment_name="Default",
):
    print(f"\n{'='*70}")
    print(f"Starting Training: {experiment_name}")
    print(f"{'='*70}")
    print(f"  lr={learning_rate}  gamma={gamma}  batch={batch_size}")
    print(f"  eps: start={epsilon_start}, end={epsilon_end}, decay={epsilon_decay}")
    print(f"  timesteps={total_timesteps}  buffer={buffer_size}")
    print(f"{'='*70}\n")

    for path in [log_file, model_save_path]:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    env = create_environment(env_name)
    policy = CnnPolicy if policy_type == "CnnPolicy" else MlpPolicy

    with open(log_file, "w", encoding="utf-8") as f:
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
        device="auto",
    )

    model.learn(total_timesteps=total_timesteps, callback=TrainingLogger(log_file))

    model.save(model_save_path)
    print(f"\nModel saved to: {model_save_path}\n")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\nTraining completed at: {datetime.now()}\n")
        f.write(f"Model saved to: {model_save_path}\n")

    return model, env, {
        "environment": env_name,
        "policy": policy_type,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "batch_size": batch_size,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay": epsilon_decay,
        "buffer_size": buffer_size,
        "total_timesteps": total_timesteps,
        "model_path": model_save_path,
    }


# ---------------------------------------------------------------------------
# Behavior summariser
# ---------------------------------------------------------------------------
def summarize_noted_behavior(log_file_path):
    if not os.path.exists(log_file_path):
        return "No log data found"

    reward_pattern = re.compile(r"Mean Reward \(last 100\):\s*(-?\d+(?:\.\d+)?)")
    length_pattern = re.compile(r"Mean Episode Length \(last 100\):\s*(-?\d+(?:\.\d+)?)")
    rewards, lengths = [], []

    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            rm = reward_pattern.search(line)
            lm = length_pattern.search(line)
            if rm:
                rewards.append(float(rm.group(1)))
            if lm:
                lengths.append(float(lm.group(1)))

    if not rewards:
        return "No metrics logged"

    first_r, last_r = rewards[0], rewards[-1]
    first_l, last_l = lengths[0], lengths[-1] if lengths else (0, 0)

    reward_trend = (
        "improving" if last_r > first_r + 0.5
        else "declining" if last_r < first_r - 0.5
        else "stable"
    )
    length_trend = (
        "increased" if last_l > first_l + 5
        else "decreased" if last_l < first_l - 5
        else "stable"
    )

    return (
        f"Reward {reward_trend} ({first_r:.2f} -> {last_r:.2f}), "
        f"episode length {length_trend}"
    )


# ---------------------------------------------------------------------------
# Track the best model across all 5 experiments
# ---------------------------------------------------------------------------
def get_best_model_path(results):
    """Return the model path with the highest final reward."""
    best = max(results, key=lambda r: float(
        r["Noted Behavior"].split("->")[-1].split(")")[0].strip()
        if "->" in r["Noted Behavior"] else -21
    ))
    return best["Model Path"], best["Experiment"]


# ---------------------------------------------------------------------------
# Run all 5 experiments
# ---------------------------------------------------------------------------
def run_1m_experiments(member_name="Alliance"):
    """Run all 5 extended experiments and append results to the shared CSV and markdown table."""

    results = []

    for config in HYPERPARAMETER_CONFIGS_1M:
        log_path = f"logs/{config['name']}_training.txt"
        model_path = f"models/{config['name']}_dqn_model.zip"

        model, env, _ = train_dqn_agent(
            env_name="ALE/Pong-v5",
            policy_type="CnnPolicy",
            total_timesteps=1000000,
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            batch_size=config["batch_size"],
            epsilon_start=config["epsilon_start"],
            epsilon_end=config["epsilon_end"],
            epsilon_decay=config["epsilon_decay"],
            model_save_path=model_path,
            log_file=log_path,
            experiment_name=config["name"],
        )

        noted_behavior = summarize_noted_behavior(log_path)
        env.close()

        results.append(
            {
                "Member Name": member_name,
                "Experiment": config["name"],
                "Learning Rate": config["learning_rate"],
                "Gamma": config["gamma"],
                "Batch Size": config["batch_size"],
                "Epsilon Start": config["epsilon_start"],
                "Epsilon End": config["epsilon_end"],
                "Epsilon Decay": config["epsilon_decay"],
                "Noted Behavior": noted_behavior,
                "Model Path": model_path,
            }
        )

    # Save best model separately
    best_path, best_name = get_best_model_path(results)
    import shutil
    shutil.copy(best_path, "best_model_alliance.zip")
    print(f"\nBest model: {best_name}")
    print(f"Saved as: best_model_alliance.zip")

    # Merge with existing results
    df_new = pd.DataFrame(results)
    csv_path = "hyperparameter_experiments.csv"
    md_path = "HYPERPARAMETER_TUNING_TABLE.md"

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["Member Name", "Experiment"], keep="last")
    else:
        df_all = df_new

    df_all.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Hyperparameter Tuning Experiments - Task 1\n\n")
        f.write("## Summary Table\n\n")
        try:
            f.write(df_all.to_markdown(index=False))
        except ImportError:
            f.write(df_all.to_string(index=False))
        f.write("\n\n## Detailed Results\n\n")
        for idx, row in df_all.iterrows():
            f.write(f"### Experiment {idx + 1}: {row['Experiment']}\n")
            for col in ["Member Name", "Learning Rate", "Gamma", "Batch Size",
                        "Epsilon Start", "Epsilon End", "Epsilon Decay",
                        "Noted Behavior", "Model Path"]:
                f.write(f"- **{col}**: {row[col]}\n")
            f.write("\n")

    print("\n" + "=" * 70)
    print("All 5 extended experiments completed!")
    print(f"Results appended to {csv_path} and {md_path}")
    print("=" * 70)
    return df_all


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DQN Extended Experiments at 1M Timesteps — Alliance"
    )
    parser.add_argument(
        "--mode",
        choices=["experiments", "final"],
        default="experiments",
        help="experiments = run all 5 configs | final = train best config only",
    )
    parser.add_argument(
        "--member",
        type=str,
        default="Alliance",
        help="Member name used in the results table",
    )
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if args.mode == "experiments":
        df = run_1m_experiments(member_name=args.member)
        print("\n" + df.to_string())
    else:
        # Train the balanced baseline as the standalone best model
        print("Training best config: 1M_Exp1_Balanced")
        cfg = next(c for c in HYPERPARAMETER_CONFIGS_1M if "Balanced" in c["name"])
        model, env, _ = train_dqn_agent(
            env_name="ALE/Pong-v5",
            policy_type="CnnPolicy",
            total_timesteps=1000000,
            learning_rate=cfg["learning_rate"],
            gamma=cfg["gamma"],
            batch_size=cfg["batch_size"],
            epsilon_start=cfg["epsilon_start"],
            epsilon_end=cfg["epsilon_end"],
            epsilon_decay=cfg["epsilon_decay"],
            model_save_path="best_model_alliance.zip",
            log_file="best_model_alliance_training.txt",
            experiment_name="Best_Model_Alliance",
        )
        env.close()
        print("\nDone. Model saved to best_model_alliance.zip")
