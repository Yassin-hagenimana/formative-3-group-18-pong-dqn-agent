# Task 1: Quick Start Guide

## Summary
Train a DQN agent to play Atari Pong using Stable Baselines3 and Gymnasium.

## Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Final Model (Recommended for testing)
```bash
python train.py --mode final
```

**Expected Output:**
- [Done] `dqn_model.zip` - Ready for use in play.py
- [Done] `final_training.txt` - Training log
- [Time] Runtime: ~15-30 minutes on GPU, ~1-2 hours on CPU

### 3. Run All 10 Hyperparameter Experiments
```bash
python train.py --mode experiments
```

**Expected Output:**
- [Done] `models/` - 10 trained models
- [Done] `logs/` - 10 training logs
- [Done] `hyperparameter_experiments.csv` - Results table
- [Done] `HYPERPARAMETER_TUNING_TABLE.md` - Detailed documentation
- [Time] Runtime: 2-5 hours on GPU, 8-12 hours on CPU
