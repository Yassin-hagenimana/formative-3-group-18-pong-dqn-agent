# Hyperparameter Tuning Experiments - Task 1

## Summary Table

| Experiment                       |   Learning Rate |   Gamma |   Batch Size |   Epsilon Start |   Epsilon End |   Epsilon Decay | Model Path                                            |
|:---------------------------------|----------------:|--------:|-------------:|----------------:|--------------:|----------------:|:------------------------------------------------------|
| Exp1_HighLR_LowGamma             |          0.001  |   0.95  |           32 |             1   |          0.05 |           0.995 | models/Exp1_HighLR_LowGamma_dqn_model.zip             |
| Exp2_LowLR_HighGamma             |          5e-05  |   0.99  |           32 |             1   |          0.05 |           0.995 | models/Exp2_LowLR_HighGamma_dqn_model.zip             |
| Exp3_LargeBatch_HighEpsilon      |          0.0001 |   0.99  |           64 |             1   |          0.1  |           0.99  | models/Exp3_LargeBatch_HighEpsilon_dqn_model.zip      |
| Exp4_SmallBatch_LowEpsilon       |          0.0001 |   0.99  |           16 |             0.5 |          0.02 |           0.98  | models/Exp4_SmallBatch_LowEpsilon_dqn_model.zip       |
| Exp5_MediumLR_MediumGamma        |          0.0003 |   0.97  |           32 |             1   |          0.05 |           0.995 | models/Exp5_MediumLR_MediumGamma_dqn_model.zip        |
| Exp6_HighLR_HighGamma_LargeBatch |          0.0005 |   0.995 |           64 |             1   |          0.01 |           0.99  | models/Exp6_HighLR_HighGamma_LargeBatch_dqn_model.zip |
| Exp7_LowLR_LowGamma_SmallBatch   |          1e-05  |   0.9   |           16 |             1   |          0.05 |           0.99  | models/Exp7_LowLR_LowGamma_SmallBatch_dqn_model.zip   |
| Exp8_BalancedConfig              |          0.0002 |   0.99  |           32 |             1   |          0.05 |           0.995 | models/Exp8_BalancedConfig_dqn_model.zip              |
| Exp9_AggressiveExploration       |          0.0001 |   0.99  |           32 |             1   |          0.1  |           0.985 | models/Exp9_AggressiveExploration_dqn_model.zip       |
| Exp10_ConservativeConfig         |          0.0001 |   0.99  |           32 |             0.9 |          0.02 |           0.998 | models/Exp10_ConservativeConfig_dqn_model.zip         |

## Detailed Results

### Experiment 1: Exp1_HighLR_LowGamma
- **Learning Rate (lr)**: 0.001
- **Gamma (γ)**: 0.95
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Model Path**: models/Exp1_HighLR_LowGamma_dqn_model.zip

### Experiment 2: Exp2_LowLR_HighGamma
- **Learning Rate (lr)**: 5e-05
- **Gamma (γ)**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Model Path**: models/Exp2_LowLR_HighGamma_dqn_model.zip

### Experiment 3: Exp3_LargeBatch_HighEpsilon
- **Learning Rate (lr)**: 0.0001
- **Gamma (γ)**: 0.99
- **Batch Size**: 64
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.1
- **Epsilon Decay**: 0.99
- **Model Path**: models/Exp3_LargeBatch_HighEpsilon_dqn_model.zip

### Experiment 4: Exp4_SmallBatch_LowEpsilon
- **Learning Rate (lr)**: 0.0001
- **Gamma (γ)**: 0.99
- **Batch Size**: 16
- **Epsilon Start**: 0.5
- **Epsilon End**: 0.02
- **Epsilon Decay**: 0.98
- **Model Path**: models/Exp4_SmallBatch_LowEpsilon_dqn_model.zip

### Experiment 5: Exp5_MediumLR_MediumGamma
- **Learning Rate (lr)**: 0.0003
- **Gamma (γ)**: 0.97
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Model Path**: models/Exp5_MediumLR_MediumGamma_dqn_model.zip

### Experiment 6: Exp6_HighLR_HighGamma_LargeBatch
- **Learning Rate (lr)**: 0.0005
- **Gamma (γ)**: 0.995
- **Batch Size**: 64
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.01
- **Epsilon Decay**: 0.99
- **Model Path**: models/Exp6_HighLR_HighGamma_LargeBatch_dqn_model.zip

### Experiment 7: Exp7_LowLR_LowGamma_SmallBatch
- **Learning Rate (lr)**: 1e-05
- **Gamma (γ)**: 0.9
- **Batch Size**: 16
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.99
- **Model Path**: models/Exp7_LowLR_LowGamma_SmallBatch_dqn_model.zip

### Experiment 8: Exp8_BalancedConfig
- **Learning Rate (lr)**: 0.0002
- **Gamma (γ)**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Model Path**: models/Exp8_BalancedConfig_dqn_model.zip

### Experiment 9: Exp9_AggressiveExploration
- **Learning Rate (lr)**: 0.0001
- **Gamma (γ)**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.1
- **Epsilon Decay**: 0.985
- **Model Path**: models/Exp9_AggressiveExploration_dqn_model.zip

### Experiment 10: Exp10_ConservativeConfig
- **Learning Rate (lr)**: 0.0001
- **Gamma (γ)**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 0.9
- **Epsilon End**: 0.02
- **Epsilon Decay**: 0.998
- **Model Path**: models/Exp10_ConservativeConfig_dqn_model.zip

