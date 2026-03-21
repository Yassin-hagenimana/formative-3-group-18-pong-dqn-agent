# Hyperparameter Tuning Experiments - Task 1

## Summary Table

| Member Name   | Experiment                             |   Learning Rate |   Gamma |   Batch Size |   Epsilon Start |   Epsilon End |   Epsilon Decay | Noted Behavior                                                          | Model Path                                                  |
|:--------------|:---------------------------------------|----------------:|--------:|-------------:|----------------:|--------------:|----------------:|:------------------------------------------------------------------------|:------------------------------------------------------------|
| Yassin        | Exp1_HighLR_LowGamma                   |          0.001  |   0.95  |           32 |             1   |          0.05 |           0.995 | Reward mostly stable (-20.26->-20.64), stable, episode length decreased | models/Exp1_HighLR_LowGamma_dqn_model.zip                   |
| Yassin        | Exp2_LowLR_HighGamma                   |          5e-05  |   0.99  |           32 |             1   |          0.05 |           0.995 | Reward mostly stable (-20.32->-20.70), stable, episode length decreased | models/Exp2_LowLR_HighGamma_dqn_model.zip                   |
| Yassin        | Exp3_LargeBatch_HighEpsilon            |          0.0001 |   0.99  |           64 |             1   |          0.1  |           0.99  | Reward mostly stable (-20.69->-20.72), stable, episode length increased | models/Exp3_LargeBatch_HighEpsilon_dqn_model.zip            |
| Yassin        | Exp4_SmallBatch_LowEpsilon             |          0.0001 |   0.99  |           16 |             0.5 |          0.02 |           0.98  | Reward mostly stable (-20.52->-20.68), stable, episode length decreased | models/Exp4_SmallBatch_LowEpsilon_dqn_model.zip             |
| Yassin        | Exp5_MediumLR_MediumGamma              |          0.0003 |   0.97  |           32 |             1   |          0.05 |           0.995 | Reward mostly stable (-20.67->-20.69), stable, episode length increased | models/Exp5_MediumLR_MediumGamma_dqn_model.zip              |
| Yassin        | Exp6_HighLR_HighGamma_LargeBatch       |          0.0005 |   0.995 |           64 |             1   |          0.01 |           0.99  | Reward mostly stable (-20.57->-20.61), stable, episode length increased | models/Exp6_HighLR_HighGamma_LargeBatch_dqn_model.zip       |
| Yassin        | Exp7_LowLR_LowGamma_SmallBatch         |          1e-05  |   0.9   |           16 |             1   |          0.05 |           0.99  | Reward mostly stable (-20.28->-20.63), stable, episode length decreased | models/Exp7_LowLR_LowGamma_SmallBatch_dqn_model.zip         |
| Yassin        | Exp8_BalancedConfig                    |          0.0002 |   0.99  |           32 |             1   |          0.05 |           0.995 | Reward mostly stable (-20.40->-20.74), stable, episode length decreased | models/Exp8_BalancedConfig_dqn_model.zip                    |
| Yassin        | Exp9_AggressiveExploration             |          0.0001 |   0.99  |           32 |             1   |          0.1  |           0.985 | Reward mostly stable (-20.88->-20.63), stable, episode length increased | models/Exp9_AggressiveExploration_dqn_model.zip             |
| Yassin        | Exp10_ConservativeConfig               |          0.0001 |   0.99  |           32 |             0.9 |          0.02 |           0.998 | Reward mostly stable (-20.77->-20.64), stable, episode length increased | models/Exp10_ConservativeConfig_dqn_model.zip               |
| Stecie        | Stecie_Exp1_LowLR_LowGamma             |          1e-05  |   0.9   |           32 |             1   |          0.05 |           0.995 | Reward mostly stable (-20.59->-20.72), stable, episode length steady    | models/Stecie_Exp1_LowLR_LowGamma_dqn_model.zip             |
| Stecie        | Stecie_Exp2_HighLR_VeryHighGamma       |          0.0005 |   0.999 |           32 |             1   |          0.05 |           0.995 | Reward mostly stable (-20.93->-20.73), stable, episode length increased | models/Stecie_Exp2_HighLR_VeryHighGamma_dqn_model.zip       |
| Stecie        | Stecie_Exp3_UltraLargeBatch            |          0.0002 |   0.99  |          256 |             1   |          0.05 |           0.995 | Reward mostly stable (-20.51->-20.80), stable, episode length steady    | models/Stecie_Exp3_UltraLargeBatch_dqn_model.zip            |
| Stecie        | Stecie_Exp4_UltraSmallBatch            |          0.0002 |   0.99  |            4 |             1   |          0.05 |           0.995 | Reward mostly stable (-20.38->-20.76), stable, episode length steady    | models/Stecie_Exp4_UltraSmallBatch_dqn_model.zip            |
| Stecie        | Stecie_Exp5_NoDecay_Epsilon            |          0.0002 |   0.99  |           32 |             1   |          0.5  |           1     | Reward mostly stable (-20.52->-20.53), stable, episode length increased | models/Stecie_Exp5_NoDecay_Epsilon_dqn_model.zip            |
| Stecie        | Stecie_Exp6_FastDecay_Short            |          0.0002 |   0.99  |           32 |             1   |          0.01 |           0.95  | Reward mostly stable (-20.56->-20.67), stable, episode length increased | models/Stecie_Exp6_FastDecay_Short_dqn_model.zip            |
| Stecie        | Stecie_Exp7_HighLR_LowGamma_SmallBatch |          0.0008 |   0.92  |            8 |             1   |          0.05 |           0.995 | Reward mostly stable (-20.92->-20.65), stable, episode length increased | models/Stecie_Exp7_HighLR_LowGamma_SmallBatch_dqn_model.zip |
| Stecie        | Stecie_Exp8_LowLR_HighGamma_LargeBatch |          1e-05  |   0.995 |          128 |             1   |          0.05 |           0.995 | Reward mostly stable (-20.75->-20.58), stable, episode length increased | models/Stecie_Exp8_LowLR_HighGamma_LargeBatch_dqn_model.zip |
| Stecie        | Stecie_Exp9_ExtremeExploration         |          0.0002 |   0.99  |           32 |             1   |          0.2  |           0.98  | Reward mostly stable (-20.86->-20.65), stable, episode length decreased | models/Stecie_Exp9_ExtremeExploration_dqn_model.zip         |
| Stecie        | Stecie_Exp10_MidRange_All              |          0.0003 |   0.97  |           64 |             0.8 |          0.03 |           0.99  | Reward mostly stable (-20.92->-20.71), stable, episode length increased | models/Stecie_Exp10_MidRange_All_dqn_model.zip              |

## Detailed Results

### Experiment 1: Exp1_HighLR_LowGamma
- **Member Name**: Yassin
- **Learning Rate (lr)**: 0.001
- **Gamma (γ)**: 0.95
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.26->-20.64), stable, episode length decreased
- **Model Path**: models/Exp1_HighLR_LowGamma_dqn_model.zip

### Experiment 2: Exp2_LowLR_HighGamma
- **Member Name**: Yassin
- **Learning Rate (lr)**: 5e-05
- **Gamma (γ)**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.32->-20.70), stable, episode length decreased
- **Model Path**: models/Exp2_LowLR_HighGamma_dqn_model.zip

### Experiment 3: Exp3_LargeBatch_HighEpsilon
- **Member Name**: Yassin
- **Learning Rate (lr)**: 0.0001
- **Gamma (γ)**: 0.99
- **Batch Size**: 64
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.1
- **Epsilon Decay**: 0.99
- **Noted Behavior**: Reward mostly stable (-20.69->-20.72), stable, episode length increased
- **Model Path**: models/Exp3_LargeBatch_HighEpsilon_dqn_model.zip

### Experiment 4: Exp4_SmallBatch_LowEpsilon
- **Member Name**: Yassin
- **Learning Rate (lr)**: 0.0001
- **Gamma (γ)**: 0.99
- **Batch Size**: 16
- **Epsilon Start**: 0.5
- **Epsilon End**: 0.02
- **Epsilon Decay**: 0.98
- **Noted Behavior**: Reward mostly stable (-20.52->-20.68), stable, episode length decreased
- **Model Path**: models/Exp4_SmallBatch_LowEpsilon_dqn_model.zip

### Experiment 5: Exp5_MediumLR_MediumGamma
- **Member Name**: Yassin
- **Learning Rate (lr)**: 0.0003
- **Gamma (γ)**: 0.97
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.67->-20.69), stable, episode length increased
- **Model Path**: models/Exp5_MediumLR_MediumGamma_dqn_model.zip

### Experiment 6: Exp6_HighLR_HighGamma_LargeBatch
- **Member Name**: Yassin
- **Learning Rate (lr)**: 0.0005
- **Gamma (γ)**: 0.995
- **Batch Size**: 64
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.01
- **Epsilon Decay**: 0.99
- **Noted Behavior**: Reward mostly stable (-20.57->-20.61), stable, episode length increased
- **Model Path**: models/Exp6_HighLR_HighGamma_LargeBatch_dqn_model.zip

### Experiment 7: Exp7_LowLR_LowGamma_SmallBatch
- **Member Name**: Yassin
- **Learning Rate (lr)**: 1e-05
- **Gamma (γ)**: 0.9
- **Batch Size**: 16
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.99
- **Noted Behavior**: Reward mostly stable (-20.28->-20.63), stable, episode length decreased
- **Model Path**: models/Exp7_LowLR_LowGamma_SmallBatch_dqn_model.zip

### Experiment 8: Exp8_BalancedConfig
- **Member Name**: Yassin
- **Learning Rate (lr)**: 0.0002
- **Gamma (γ)**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.40->-20.74), stable, episode length decreased
- **Model Path**: models/Exp8_BalancedConfig_dqn_model.zip

### Experiment 9: Exp9_AggressiveExploration
- **Member Name**: Yassin
- **Learning Rate (lr)**: 0.0001
- **Gamma (γ)**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.1
- **Epsilon Decay**: 0.985
- **Noted Behavior**: Reward mostly stable (-20.88->-20.63), stable, episode length increased
- **Model Path**: models/Exp9_AggressiveExploration_dqn_model.zip

### Experiment 10: Exp10_ConservativeConfig
- **Member Name**: Yassin
- **Learning Rate (lr)**: 0.0001
- **Gamma (γ)**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 0.9
- **Epsilon End**: 0.02
- **Epsilon Decay**: 0.998
- **Noted Behavior**: Reward mostly stable (-20.77->-20.64), stable, episode length increased
- **Model Path**: models/Exp10_ConservativeConfig_dqn_model.zip

### Experiment 11: Stecie_Exp1_LowLR_LowGamma
- **Member Name**: Stecie
- **Learning Rate (lr)**: 1e-05
- **Gamma (γ)**: 0.9
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.59->-20.72), stable, episode length steady
- **Model Path**: models/Stecie_Exp1_LowLR_LowGamma_dqn_model.zip

### Experiment 12: Stecie_Exp2_HighLR_VeryHighGamma
- **Member Name**: Stecie
- **Learning Rate (lr)**: 0.0005
- **Gamma (γ)**: 0.999
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.93->-20.73), stable, episode length increased
- **Model Path**: models/Stecie_Exp2_HighLR_VeryHighGamma_dqn_model.zip

### Experiment 13: Stecie_Exp3_UltraLargeBatch
- **Member Name**: Stecie
- **Learning Rate (lr)**: 0.0002
- **Gamma (γ)**: 0.99
- **Batch Size**: 256
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.51->-20.80), stable, episode length steady
- **Model Path**: models/Stecie_Exp3_UltraLargeBatch_dqn_model.zip

### Experiment 14: Stecie_Exp4_UltraSmallBatch
- **Member Name**: Stecie
- **Learning Rate (lr)**: 0.0002
- **Gamma (γ)**: 0.99
- **Batch Size**: 4
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.38->-20.76), stable, episode length steady
- **Model Path**: models/Stecie_Exp4_UltraSmallBatch_dqn_model.zip

### Experiment 15: Stecie_Exp5_NoDecay_Epsilon
- **Member Name**: Stecie
- **Learning Rate (lr)**: 0.0002
- **Gamma (γ)**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.5
- **Epsilon Decay**: 1.0
- **Noted Behavior**: Reward mostly stable (-20.52->-20.53), stable, episode length increased
- **Model Path**: models/Stecie_Exp5_NoDecay_Epsilon_dqn_model.zip

### Experiment 16: Stecie_Exp6_FastDecay_Short
- **Member Name**: Stecie
- **Learning Rate (lr)**: 0.0002
- **Gamma (γ)**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.01
- **Epsilon Decay**: 0.95
- **Noted Behavior**: Reward mostly stable (-20.56->-20.67), stable, episode length increased
- **Model Path**: models/Stecie_Exp6_FastDecay_Short_dqn_model.zip

### Experiment 17: Stecie_Exp7_HighLR_LowGamma_SmallBatch
- **Member Name**: Stecie
- **Learning Rate (lr)**: 0.0008
- **Gamma (γ)**: 0.92
- **Batch Size**: 8
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.92->-20.65), stable, episode length increased
- **Model Path**: models/Stecie_Exp7_HighLR_LowGamma_SmallBatch_dqn_model.zip

### Experiment 18: Stecie_Exp8_LowLR_HighGamma_LargeBatch
- **Member Name**: Stecie
- **Learning Rate (lr)**: 1e-05
- **Gamma (γ)**: 0.995
- **Batch Size**: 128
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.75->-20.58), stable, episode length increased
- **Model Path**: models/Stecie_Exp8_LowLR_HighGamma_LargeBatch_dqn_model.zip

### Experiment 19: Stecie_Exp9_ExtremeExploration
- **Member Name**: Stecie
- **Learning Rate (lr)**: 0.0002
- **Gamma (γ)**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.2
- **Epsilon Decay**: 0.98
- **Noted Behavior**: Reward mostly stable (-20.86->-20.65), stable, episode length decreased
- **Model Path**: models/Stecie_Exp9_ExtremeExploration_dqn_model.zip

### Experiment 20: Stecie_Exp10_MidRange_All
- **Member Name**: Stecie
- **Learning Rate (lr)**: 0.0003
- **Gamma (γ)**: 0.97
- **Batch Size**: 64
- **Epsilon Start**: 0.8
- **Epsilon End**: 0.03
- **Epsilon Decay**: 0.99
- **Noted Behavior**: Reward mostly stable (-20.92->-20.71), stable, episode length increased
- **Model Path**: models/Stecie_Exp10_MidRange_All_dqn_model.zip

