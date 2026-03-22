# Formative 3: DQN Atari Agent - Group 18

Implementation of a Deep Q-Network (DQN) agent for **Atari Pong** using Stable Baselines3 and Gymnasium. This project includes comprehensive training, hyperparameter tuning experiments across all group members, and agent evaluation.

## Project Overview

This is a group project implementing a reinforcement learning agent using Deep Q-Networks (DQN) to play Atari Pong. The project consists of three main components:

1. **Task 1**; Training Scripts (`train.py`, `train_alliance.py`, `train_alliance_1M.py`)
2. **Task 2**; Playing Script (`play.py`)
3. **Task 3**; Group Presentation (10 minutes)

## Group Members

- **Yassin**; Training script (`train.py`), 10 hyperparameter experiments at 50k timesteps
- **Alliance**; Training scripts (`train_alliance.py`, `train_alliance_1M.py`), 10 hyperparameter experiments at 50k timesteps + 3 extended experiments at 1M timesteps, `play.py`
- **Stecie**; 10 hyperparameter experiments at 50k timesteps

---

## Agent Gameplay Demo

 <video src="hhttps://github.com/user-attachments/assets/5707258c-03e6-4ef0-bf0e-5b200d583191" controls width="800"></video>

To run the agent yourself: `python play.py`

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Validate Setup
```bash
python validate_setup.py
```

### 3. Run the Trained Agent
```bash
python play.py
```

### 4. Train Yassin's Final Model
```bash
python train.py --mode final
```

### 5. Run Alliance's Hyperparameter Experiments (50k timesteps)
```bash
python train_alliance.py --mode experiments
```

### 6. Run Alliance's Extended Experiments (1M timesteps)
```bash
python train_alliance_1M.py --mode experiments
```

---

## Best Model

The best performing model is **1M_Exp3_XLBatch** trained at 1,000,000 timesteps using a large batch size of 128 for stable gradient updates. It achieved a mean evaluation reward of **-0.40** across 5 episodes, with a best episode reward of **+10.0** significantly outperforming all 50k timestep models which remained around -20.

Saved as `dqn_model.zip` in the project root.

| Config | Timesteps | Batch Size | Mean Reward |
|--------|-----------|------------|-------------|
| 1M_Exp3_XLBatch | 1,000,000 | 128 | -0.40 |
| 1M_Exp1_Balanced | 1,000,000 | 32 | ~-10.53 |
| 1M_Exp2_LowGamma | 1,000,000 | 32 | ~-13.37 |
| Yassin Final Model | 100,000 | 32 | ~-19.48 |

---

### Head-to-Head Evaluation (5 episodes each, greedy policy)

**Using the model trained on 1 million timesteps**

| Episode | Reward | Steps |
|---------|--------|-------|
| 1 | +1.0 | 1276 |
| 2 | -5.0 | 1147 |
| 3 | -5.0 | 1044 |
| 4 | +10.0 | 771 |
| 5 | -3.0 | 990 |
| **Mean** | **-0.40** | **1046** |

**Using a model trained on 100k timesteps**

| Episode | Reward | Steps |
|---------|--------|-------|
| 1 | -21.0 | 184 |
| 2 | -21.0 | 188 |
| 3 | -21.0 | 184 |
| 4 | -21.0 | 186 |
| 5 | -21.0 | 187 |
| **Mean** | **-21.00** | **186** |

The difference is stark, the 100k model loses every single point in every episode (worst possible score of -21), while the 1M model competes effectively against the built-in Atari bot, winning episode 4 by 10 points. This directly demonstrates the impact of training duration on agent performance. The very low step count in 100k model episodes (around 184) also reveals the agent is not moving meaningfully, the game ends quickly because it never returns the ball. The best model averages over 1000 steps per episode, showing active and competitive gameplay.

---

## Hyperparameter Tuning

A total of **33 experiments** were conducted across all three group members. Full results are available in `hyperparameter_experiments.csv` and `HYPERPARAMETER_TUNING_TABLE.md`.

### Learning Rate Impact
- **Very high (5e-3)**: Causes instability, large weight updates overshoot good solutions
- **High (1e-3)**: Fast convergence but potentially unstable
- **Medium (2e-4)**: Good balance between speed and stability
- **Low (1e-5)**: Slow but stable learning
- **Very low (1e-5 with high gamma)**: Near-zero improvement within 50k steps

### Gamma (Discount Factor) Impact
- **Very high (0.999)**: Near-undiscounted returns, values long-term outcomes maximally
- **High (0.99+)**: Values future rewards, supports long-term planning in Pong
- **Medium (0.97)**: Balanced temporal credit assignment
- **Low (0.92)**: Short-sighted, misses multi-step reward sequences
- **Very low (0.80)**: Heavy short-termism, significantly harms Pong performance

### Batch Size Impact
- **Ultra large (256)**: Maximum gradient stability, very slow iteration
- **Extra large (128)**: Strong stability benefit, best performing config at 1M timesteps
- **Large (64)**: More stable updates, less variance
- **Standard (32)**: Good balance between stability and speed
- **Small (16)**: Faster updates, more variance
- **Mini (8)**: Very noisy updates, erratic behavior
- **Ultra small (4)**: Extremely noisy, poor convergence

### Epsilon (Exploration) Impact
- **Immediate exploit (start=0.2)**: Agent locks into bad habits before learning anything useful
- **No decay (end=0.5, decay=1.0)**: Agent never fully exploits, permanent performance ceiling
- **Aggressive decay (0.95–0.97)**: Quick transition to exploitation, risks committing too early
- **Standard decay (0.995)**: Balanced exploration-exploitation tradeoff
- **Conservative decay (0.998–0.999)**: Sustained exploration throughout training
- **Near-zero end (0.005)**: Agent commits almost fully to exploitation by end of training

### Key Finding: Timesteps Matter Most
The most significant insight from the experiments is the effect of training duration. All 50k timestep experiments across all three members produced rewards stuck around -20 regardless of hyperparameter configuration. The 1M timestep experiments showed dramatically improved performance, with the best model achieving a mean reward of -0.40. This demonstrates that for Atari Pong, sufficient training time is a prerequisite for hyperparameter differences to become meaningful.

---

## Hyperparameter Tuning Table

See `HYPERPARAMETER_TUNING_TABLE.md` for the full combined table of all 33 experiments across all group members.

---

## Task 2: Playing the Agent

The `play.py` script loads the best trained model and evaluates it using a **greedy policy**; the agent always selects the action with the highest Q-value with no random exploration (epsilon = 0).

```bash
# Default: loads dqn_model.zip, plays 5 episodes with rendering
python play.py

# Play fewer episodes for a quick demo
python play.py --episodes 2

# Use a specific model
python play.py --model models/1M_Exp3_XLBatch_dqn_model.zip

# Run without rendering
python play.py --no-render
```

https://youtu.be/IAuxIun2eqk

---

## Output Structure

```
dqn_model.zip                        ← Best trained model (1M_Exp3_XLBatch)
final_training.txt                   ← Yassin's final model training log
hyperparameter_experiments.csv       ← All 33 experiments combined
HYPERPARAMETER_TUNING_TABLE.md       ← Formatted results table

models/
├── Exp1_HighLR_LowGamma_dqn_model.zip          ← Yassin's experiments
├── ... (9 more Yassin models)
├── Exp1_VeryHighLR_FastDecay_dqn_model.zip      ← Alliance's 50k experiments
├── ... (9 more Alliance 50k models)
├── Stecie_Exp1_LowLR_LowGamma_dqn_model.zip    ← Stecie's experiments
├── ... (9 more Stecie models)
├── 1M_Exp1_Balanced_dqn_model.zip              ← Alliance's 1M experiments
├── 1M_Exp2_LowGamma_dqn_model.zip
└── 1M_Exp3_XLBatch_dqn_model.zip

logs/
├── (training logs for all experiments)
```

---

## System Requirements

- **Python**: 3.8+
- **RAM**: 4GB+ (8GB recommended)
- **GPU**: CUDA-compatible (optional, for faster training)
- **VRAM**: 4GB+ (if using GPU)

---

## Performance Benchmarks

| Script | Timesteps | CPU Time |
|--------|-----------|----------|
| `train.py --mode final` | 100k | 1-2 hrs |
| `train.py --mode experiments` | 50k each | 8-12 hrs total |
| `train_alliance.py --mode experiments` | 50k each | 4-7 hrs total |
| `train_alliance_1M.py --mode experiments` | 1M each | ~5 hrs each |

---

## Troubleshooting

### ModuleNotFoundError: No module named 'gymnasium'
```bash
# Make sure your virtual environment is activated first
.venv\Scripts\activate
pip install -r requirements.txt
```

### CUDA Out of Memory
```bash
# Training automatically falls back to CPU if CUDA is unavailable
python train.py --mode final
```

### Missing Atari ROMs
```bash
pip install ale-py
```

### Slow Training
```bash
# Reduce timesteps for testing
python train_alliance.py --mode experiments --timesteps 10000
```

---

## Additional Resources

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [DQN Paper - Mnih et al., 2015](https://www.nature.com/articles/nature14236)
- [Atari Environments](https://gymnasium.farama.org/environments/atari/)

---

## License

This project is created as part of the Formative 3 assignment for Deep Reinforcement Learning.
