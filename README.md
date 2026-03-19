# Formative 3: DQN Atari Agent - Group 18

Implementation of a Deep Q-Network (DQN) agent for **Atari Pong** using Stable Baselines3 and Gymnasium. This project includes comprehensive training, hyperparameter tuning experiments, and agent evaluation.

## Project Overview

This is a group project implementing a reinforcement learning agent using Deep Q-Networks (DQN) to play Atari Pong. The project consists of three main components:

1. **Task 1**  Training Script (`train.py`)
2. **Task 2**  Playing Script (`play.py`)
3. **Task 3**  Group Presentation (10 minutes)


## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Validate Setup
```bash
python validate_setup.py
```

### 3. Train Final Model
```bash
python train.py --mode final
```

Output: `dqn_model.zip` (ready for play.py)

### 4. Run All 10 Experiments
```bash
python train.py --mode experiments
```

Outputs:
- `models/` - 10 trained models
- `logs/` - 10 training logs
- `hyperparameter_experiments.csv` - Results table
- `HYPERPARAMETER_TUNING_TABLE.md` - Analysis

---

## Hyperparameter Tuning Analysis

### Learning Rate Impact
- **High (1e-3)**: Fast convergence but potentially unstable
- **Medium (2e-4)**: Good balance between speed and stability
- **Low (1e-5)**: Slow but stable learning

### Gamma (Discount Factor) Impact
- **High (0.99+)**: Values future rewards, long-term planning
- **Medium (0.97)**: Balanced temporal credit assignment
- **Low (0.9)**: Emphasizes immediate rewards

### Batch Size Impact
- **Large (64)**: More stable updates, less variance
- **Normal (32)**: Good balance
- **Small (16)**: Faster updates, more variance

### Epsilon (Exploration) Impact
- **Aggressive decay (0.985)**: Quick transition to exploitation
- **Standard decay (0.995)**: Balanced exploration-exploitation
- **Conservative decay (0.998)**: Sustained exploration

---

## Training Usage Examples

### Train with Default Configuration
```bash
python train.py --mode final
```

### Train with MlpPolicy Instead
```bash
python train.py --mode final --policy MlpPolicy
```

### Custom Training Parameters
```bash
python train.py --mode final --timesteps 50000
```

### Run Hyperparameter Experiments
```bash
python train.py --mode experiments --policy CnnPolicy
```

### Help
```bash
python train.py --help
```

---

## Output Structure After Training

### Final Model Training
```
dqn_model.zip              ← Trained agent (for play.py)
final_training.txt         ← Training log
```

### Hyperparameter Experiments
```
models/
├── Exp1_HighLR_LowGamma_dqn_model.zip
├── Exp2_LowLR_HighGamma_dqn_model.zip
├── ... (8 more)
└── Exp10_ConservativeConfig_dqn_model.zip

logs/
├── Exp1_HighLR_LowGamma_training.txt
├── Exp2_LowLR_HighGamma_training.txt
├── ... (8 more)
└── Exp10_ConservativeConfig_training.txt

hyperparameter_experiments.csv           ← Results table
HYPERPARAMETER_TUNING_TABLE.md          ← Detailed analysis
```

## System Requirements

- **Python**: 3.8+
- **RAM**: 4GB+ (8GB recommended for GPU)
- **GPU**: CUDA-compatible (optional, for faster training)
- **VRAM**: 4GB+ (if using GPU)

---

## Performance Benchmarks

| Mode | Timesteps | GPU | CPU |
|------|-----------|-----|-----|
| Final Model | 100k | 15-30 min | 1-2 hrs |
| Experiments | 50k each | 2-5 hrs | 8-12 hrs |

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Use CPU instead
python train.py --mode final  # device will be auto-set to CPU
```

### Missing Gymnasium
```bash
pip install gymnasium gymnasium[atari]
```

### Missing Atari ROMs
```bash
pip install atari-py
python -m atari_py.import_roms ~/Downloads/Roms
```

### Slow Training
```bash
# Reduce timesteps for testing
python train.py --mode final --timesteps 10000
```

---

## Additional Resources

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [DQN Paper - Mnih et al., 2015](https://www.nature.com/articles/nature14236)
- [Atari Environments](https://gymnasium.farama.org/environments/atari/)



## License

This project is created as part of the Formative 3 assignment for Deep Reinforcement Learning.
