"""
Hyperparameter Configuration File
Modify these values to easily test different configurations without editing train.py
"""

# Environment Configuration
ENVIRONMENT = 'ALE/Pong-v5'  # Atari environment name

# Training Configuration
TOTAL_TIMESTEPS = 100000      # Total training steps
POLICY_TYPE = 'CnnPolicy'     # 'CnnPolicy' or 'MlpPolicy'

# Core Hyperparameters
LEARNING_RATE = 2e-4          # Learning rate for optimizer (1e-5 to 5e-4)
GAMMA = 0.99                  # Discount factor (0.9 to 0.995)
BATCH_SIZE = 32               # Batch size for training (16 to 64)

# Exploration Parameters (Epsilon-Greedy)
EPSILON_START = 1.0           # Initial exploration rate (0.5 to 1.0)
EPSILON_END = 0.05            # Final exploration rate (0.01 to 0.1)
EPSILON_DECAY = 0.995         # Decay rate (0.98 to 0.998)

# File Paths
MODEL_SAVE_PATH = 'dqn_model.zip'
LOG_FILE = 'training_log.txt'

# Experiment Configurations for Hyperparameter Tuning
# Each tuple: (name, lr, gamma, batch_size, eps_start, eps_end, eps_decay)
HYPERPARAMETER_CONFIGS = [
    ('Exp1_HighLR_LowGamma', 1e-3, 0.95, 32, 1.0, 0.05, 0.995),
    ('Exp2_LowLR_HighGamma', 5e-5, 0.99, 32, 1.0, 0.05, 0.995),
    ('Exp3_LargeBatch_HighEpsilon', 1e-4, 0.99, 64, 1.0, 0.1, 0.99),
    ('Exp4_SmallBatch_LowEpsilon', 1e-4, 0.99, 16, 0.5, 0.02, 0.98),
    ('Exp5_MediumLR_MediumGamma', 3e-4, 0.97, 32, 1.0, 0.05, 0.995),
    ('Exp6_HighLR_HighGamma_LargeBatch', 5e-4, 0.995, 64, 1.0, 0.01, 0.99),
    ('Exp7_LowLR_LowGamma_SmallBatch', 1e-5, 0.9, 16, 1.0, 0.05, 0.99),
    ('Exp8_BalancedConfig', 2e-4, 0.99, 32, 1.0, 0.05, 0.995),
    ('Exp9_AggressiveExploration', 1e-4, 0.99, 32, 1.0, 0.1, 0.985),
    ('Exp10_ConservativeConfig', 1e-4, 0.99, 32, 0.9, 0.02, 0.998),
]

# Experiment Notes
EXPERIMENT_NOTES = {
    'Exp1_HighLR_LowGamma': 'High learning rate + low discount factor for fast, short-term learning',
    'Exp2_LowLR_HighGamma': 'Low learning rate + high discount for stable, long-term learning',
    'Exp3_LargeBatch_HighEpsilon': 'Large batch for stable updates + higher exploration',
    'Exp4_SmallBatch_LowEpsilon': 'Small batch for rapid updates + low exploration',
    'Exp5_MediumLR_MediumGamma': 'Balanced medium configuration',
    'Exp6_HighLR_HighGamma_LargeBatch': 'Aggressive learning + maximum discount + large batch',
    'Exp7_LowLR_LowGamma_SmallBatch': 'Conservative learning + low discount + small batch',
    'Exp8_BalancedConfig': 'Standard balanced configuration',
    'Exp9_AggressiveExploration': 'Emphasis on sustained exploration',
    'Exp10_ConservativeConfig': 'Conservative exploration decay',
}
