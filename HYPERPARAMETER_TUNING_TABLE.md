# Hyperparameter Tuning Experiments - Task 1

## Summary Table

Member Name                             Experiment  Learning Rate  Gamma  Batch Size  Epsilon Start  Epsilon End  Epsilon Decay                                                          Noted Behavior                                                  Model Path
     Yassin                   Exp1_HighLR_LowGamma        0.00100  0.950          32            1.0        0.050          0.995 Reward mostly stable (-20.26->-20.64), stable, episode length decreased                   models/Exp1_HighLR_LowGamma_dqn_model.zip
     Yassin                   Exp2_LowLR_HighGamma        0.00005  0.990          32            1.0        0.050          0.995 Reward mostly stable (-20.32->-20.70), stable, episode length decreased                   models/Exp2_LowLR_HighGamma_dqn_model.zip
     Yassin            Exp3_LargeBatch_HighEpsilon        0.00010  0.990          64            1.0        0.100          0.990 Reward mostly stable (-20.69->-20.72), stable, episode length increased            models/Exp3_LargeBatch_HighEpsilon_dqn_model.zip
     Yassin             Exp4_SmallBatch_LowEpsilon        0.00010  0.990          16            0.5        0.020          0.980 Reward mostly stable (-20.52->-20.68), stable, episode length decreased             models/Exp4_SmallBatch_LowEpsilon_dqn_model.zip
     Yassin              Exp5_MediumLR_MediumGamma        0.00030  0.970          32            1.0        0.050          0.995 Reward mostly stable (-20.67->-20.69), stable, episode length increased              models/Exp5_MediumLR_MediumGamma_dqn_model.zip
     Yassin       Exp6_HighLR_HighGamma_LargeBatch        0.00050  0.995          64            1.0        0.010          0.990 Reward mostly stable (-20.57->-20.61), stable, episode length increased       models/Exp6_HighLR_HighGamma_LargeBatch_dqn_model.zip
     Yassin         Exp7_LowLR_LowGamma_SmallBatch        0.00001  0.900          16            1.0        0.050          0.990 Reward mostly stable (-20.28->-20.63), stable, episode length decreased         models/Exp7_LowLR_LowGamma_SmallBatch_dqn_model.zip
     Yassin                    Exp8_BalancedConfig        0.00020  0.990          32            1.0        0.050          0.995 Reward mostly stable (-20.40->-20.74), stable, episode length decreased                    models/Exp8_BalancedConfig_dqn_model.zip
     Yassin             Exp9_AggressiveExploration        0.00010  0.990          32            1.0        0.100          0.985 Reward mostly stable (-20.88->-20.63), stable, episode length increased             models/Exp9_AggressiveExploration_dqn_model.zip
     Yassin               Exp10_ConservativeConfig        0.00010  0.990          32            0.9        0.020          0.998 Reward mostly stable (-20.77->-20.64), stable, episode length increased               models/Exp10_ConservativeConfig_dqn_model.zip
   Alliance              Exp1_VeryHighLR_FastDecay        0.00500  0.990          32            1.0        0.050          0.970                  Reward stable (-20.75 → -20.62), episode length stable              models/Exp1_VeryHighLR_FastDecay_dqn_model.zip
   Alliance           Exp2_VeryLowGamma_LargeBatch        0.00020  0.800          64            1.0        0.050          0.995                  Reward stable (-21.00 → -20.66), episode length stable           models/Exp2_VeryLowGamma_LargeBatch_dqn_model.zip
   Alliance                     Exp3_XLBatch_StdLR        0.00010  0.990         128            1.0        0.050          0.995               Reward stable (-20.50 → -20.62), episode length decreased                     models/Exp3_XLBatch_StdLR_dqn_model.zip
   Alliance           Exp4_MiniBatch_VeryFastDecay        0.00010  0.990           8            1.0        0.050          0.960               Reward stable (-21.00 → -20.67), episode length increased           models/Exp4_MiniBatch_VeryFastDecay_dqn_model.zip
   Alliance                Exp5_MaxGamma_VeryLowLR        0.00001  0.999          32            1.0        0.050          0.995                  Reward stable (-20.50 → -20.78), episode length stable                models/Exp5_MaxGamma_VeryLowLR_dqn_model.zip
   Alliance       Exp6_ZeroWarmup_ImmediateExploit        0.00020  0.990          32            0.2        0.050          0.990                  Reward stable (-21.00 → -20.63), episode length stable       models/Exp6_ZeroWarmup_ImmediateExploit_dqn_model.zip
   Alliance           Exp7_HighLR_XLBatch_LowGamma        0.00100  0.920         128            1.0        0.050          0.990            Reward declining (-20.00 → -20.62), episode length decreased           models/Exp7_HighLR_XLBatch_LowGamma_dqn_model.zip
   Alliance                Exp8_LowEnd_VeryHighEps        0.00010  0.990          32            1.0        0.200          0.995               Reward stable (-20.75 → -20.71), episode length decreased                models/Exp8_LowEnd_VeryHighEps_dqn_model.zip
   Alliance         Exp9_HighLR_MidGamma_SlowDecay        0.00080  0.960          32            1.0        0.050          0.999               Reward stable (-20.75 → -20.72), episode length decreased         models/Exp9_HighLR_MidGamma_SlowDecay_dqn_model.zip
   Alliance   Exp10_SmallBatch_HighGamma_LowEpsEnd        0.00030  0.995          16            0.8        0.005          0.990               Reward stable (-20.75 → -20.55), episode length increased   models/Exp10_SmallBatch_HighGamma_LowEpsEnd_dqn_model.zip
     Stecie             Stecie_Exp1_LowLR_LowGamma        0.00001  0.900          32            1.0        0.050          0.995    Reward mostly stable (-20.59->-20.72), stable, episode length steady             models/Stecie_Exp1_LowLR_LowGamma_dqn_model.zip
     Stecie       Stecie_Exp2_HighLR_VeryHighGamma        0.00050  0.999          32            1.0        0.050          0.995 Reward mostly stable (-20.93->-20.73), stable, episode length increased       models/Stecie_Exp2_HighLR_VeryHighGamma_dqn_model.zip
     Stecie            Stecie_Exp3_UltraLargeBatch        0.00020  0.990         256            1.0        0.050          0.995    Reward mostly stable (-20.51->-20.80), stable, episode length steady            models/Stecie_Exp3_UltraLargeBatch_dqn_model.zip
     Stecie            Stecie_Exp4_UltraSmallBatch        0.00020  0.990           4            1.0        0.050          0.995    Reward mostly stable (-20.38->-20.76), stable, episode length steady            models/Stecie_Exp4_UltraSmallBatch_dqn_model.zip
     Stecie            Stecie_Exp5_NoDecay_Epsilon        0.00020  0.990          32            1.0        0.500          1.000 Reward mostly stable (-20.52->-20.53), stable, episode length increased            models/Stecie_Exp5_NoDecay_Epsilon_dqn_model.zip
     Stecie            Stecie_Exp6_FastDecay_Short        0.00020  0.990          32            1.0        0.010          0.950 Reward mostly stable (-20.56->-20.67), stable, episode length increased            models/Stecie_Exp6_FastDecay_Short_dqn_model.zip
     Stecie Stecie_Exp7_HighLR_LowGamma_SmallBatch        0.00080  0.920           8            1.0        0.050          0.995 Reward mostly stable (-20.92->-20.65), stable, episode length increased models/Stecie_Exp7_HighLR_LowGamma_SmallBatch_dqn_model.zip
     Stecie Stecie_Exp8_LowLR_HighGamma_LargeBatch        0.00001  0.995         128            1.0        0.050          0.995 Reward mostly stable (-20.75->-20.58), stable, episode length increased models/Stecie_Exp8_LowLR_HighGamma_LargeBatch_dqn_model.zip
     Stecie         Stecie_Exp9_ExtremeExploration        0.00020  0.990          32            1.0        0.200          0.980 Reward mostly stable (-20.86->-20.65), stable, episode length decreased         models/Stecie_Exp9_ExtremeExploration_dqn_model.zip
     Stecie              Stecie_Exp10_MidRange_All        0.00030  0.970          64            0.8        0.030          0.990 Reward mostly stable (-20.92->-20.71), stable, episode length increased              models/Stecie_Exp10_MidRange_All_dqn_model.zip
   Alliance                       1M_Exp1_Balanced        0.00010  0.990          32            1.0        0.050          0.995           Reward improving (-20.50 -> -10.53), episode length increased                       models/1M_Exp1_Balanced_dqn_model.zip
   Alliance                       1M_Exp2_LowGamma        0.00010  0.800          32            1.0        0.050          0.995           Reward improving (-20.50 -> -13.37), episode length increased                       models/1M_Exp2_LowGamma_dqn_model.zip
   Alliance                        1M_Exp3_XLBatch        0.00010  0.990         128            1.0        0.050          0.995           Reward improving (-20.50 -> -10.03), episode length increased                        models/1M_Exp3_XLBatch_dqn_model.zip

## Detailed Results

### Experiment 1: Exp1_HighLR_LowGamma
- **Member Name**: Yassin
- **Learning Rate**: 0.001
- **Gamma**: 0.95
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.26->-20.64), stable, episode length decreased
- **Model Path**: models/Exp1_HighLR_LowGamma_dqn_model.zip

### Experiment 2: Exp2_LowLR_HighGamma
- **Member Name**: Yassin
- **Learning Rate**: 5e-05
- **Gamma**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.32->-20.70), stable, episode length decreased
- **Model Path**: models/Exp2_LowLR_HighGamma_dqn_model.zip

### Experiment 3: Exp3_LargeBatch_HighEpsilon
- **Member Name**: Yassin
- **Learning Rate**: 0.0001
- **Gamma**: 0.99
- **Batch Size**: 64
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.1
- **Epsilon Decay**: 0.99
- **Noted Behavior**: Reward mostly stable (-20.69->-20.72), stable, episode length increased
- **Model Path**: models/Exp3_LargeBatch_HighEpsilon_dqn_model.zip

### Experiment 4: Exp4_SmallBatch_LowEpsilon
- **Member Name**: Yassin
- **Learning Rate**: 0.0001
- **Gamma**: 0.99
- **Batch Size**: 16
- **Epsilon Start**: 0.5
- **Epsilon End**: 0.02
- **Epsilon Decay**: 0.98
- **Noted Behavior**: Reward mostly stable (-20.52->-20.68), stable, episode length decreased
- **Model Path**: models/Exp4_SmallBatch_LowEpsilon_dqn_model.zip

### Experiment 5: Exp5_MediumLR_MediumGamma
- **Member Name**: Yassin
- **Learning Rate**: 0.0003
- **Gamma**: 0.97
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.67->-20.69), stable, episode length increased
- **Model Path**: models/Exp5_MediumLR_MediumGamma_dqn_model.zip

### Experiment 6: Exp6_HighLR_HighGamma_LargeBatch
- **Member Name**: Yassin
- **Learning Rate**: 0.0005
- **Gamma**: 0.995
- **Batch Size**: 64
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.01
- **Epsilon Decay**: 0.99
- **Noted Behavior**: Reward mostly stable (-20.57->-20.61), stable, episode length increased
- **Model Path**: models/Exp6_HighLR_HighGamma_LargeBatch_dqn_model.zip

### Experiment 7: Exp7_LowLR_LowGamma_SmallBatch
- **Member Name**: Yassin
- **Learning Rate**: 1e-05
- **Gamma**: 0.9
- **Batch Size**: 16
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.99
- **Noted Behavior**: Reward mostly stable (-20.28->-20.63), stable, episode length decreased
- **Model Path**: models/Exp7_LowLR_LowGamma_SmallBatch_dqn_model.zip

### Experiment 8: Exp8_BalancedConfig
- **Member Name**: Yassin
- **Learning Rate**: 0.0002
- **Gamma**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.40->-20.74), stable, episode length decreased
- **Model Path**: models/Exp8_BalancedConfig_dqn_model.zip

### Experiment 9: Exp9_AggressiveExploration
- **Member Name**: Yassin
- **Learning Rate**: 0.0001
- **Gamma**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.1
- **Epsilon Decay**: 0.985
- **Noted Behavior**: Reward mostly stable (-20.88->-20.63), stable, episode length increased
- **Model Path**: models/Exp9_AggressiveExploration_dqn_model.zip

### Experiment 10: Exp10_ConservativeConfig
- **Member Name**: Yassin
- **Learning Rate**: 0.0001
- **Gamma**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 0.9
- **Epsilon End**: 0.02
- **Epsilon Decay**: 0.998
- **Noted Behavior**: Reward mostly stable (-20.77->-20.64), stable, episode length increased
- **Model Path**: models/Exp10_ConservativeConfig_dqn_model.zip

### Experiment 11: Exp1_VeryHighLR_FastDecay
- **Member Name**: Alliance
- **Learning Rate**: 0.005
- **Gamma**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.97
- **Noted Behavior**: Reward stable (-20.75 → -20.62), episode length stable
- **Model Path**: models/Exp1_VeryHighLR_FastDecay_dqn_model.zip

### Experiment 12: Exp2_VeryLowGamma_LargeBatch
- **Member Name**: Alliance
- **Learning Rate**: 0.0002
- **Gamma**: 0.8
- **Batch Size**: 64
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward stable (-21.00 → -20.66), episode length stable
- **Model Path**: models/Exp2_VeryLowGamma_LargeBatch_dqn_model.zip

### Experiment 13: Exp3_XLBatch_StdLR
- **Member Name**: Alliance
- **Learning Rate**: 0.0001
- **Gamma**: 0.99
- **Batch Size**: 128
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward stable (-20.50 → -20.62), episode length decreased
- **Model Path**: models/Exp3_XLBatch_StdLR_dqn_model.zip

### Experiment 14: Exp4_MiniBatch_VeryFastDecay
- **Member Name**: Alliance
- **Learning Rate**: 0.0001
- **Gamma**: 0.99
- **Batch Size**: 8
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.96
- **Noted Behavior**: Reward stable (-21.00 → -20.67), episode length increased
- **Model Path**: models/Exp4_MiniBatch_VeryFastDecay_dqn_model.zip

### Experiment 15: Exp5_MaxGamma_VeryLowLR
- **Member Name**: Alliance
- **Learning Rate**: 1e-05
- **Gamma**: 0.999
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward stable (-20.50 → -20.78), episode length stable
- **Model Path**: models/Exp5_MaxGamma_VeryLowLR_dqn_model.zip

### Experiment 16: Exp6_ZeroWarmup_ImmediateExploit
- **Member Name**: Alliance
- **Learning Rate**: 0.0002
- **Gamma**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 0.2
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.99
- **Noted Behavior**: Reward stable (-21.00 → -20.63), episode length stable
- **Model Path**: models/Exp6_ZeroWarmup_ImmediateExploit_dqn_model.zip

### Experiment 17: Exp7_HighLR_XLBatch_LowGamma
- **Member Name**: Alliance
- **Learning Rate**: 0.001
- **Gamma**: 0.92
- **Batch Size**: 128
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.99
- **Noted Behavior**: Reward declining (-20.00 → -20.62), episode length decreased
- **Model Path**: models/Exp7_HighLR_XLBatch_LowGamma_dqn_model.zip

### Experiment 18: Exp8_LowEnd_VeryHighEps
- **Member Name**: Alliance
- **Learning Rate**: 0.0001
- **Gamma**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.2
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward stable (-20.75 → -20.71), episode length decreased
- **Model Path**: models/Exp8_LowEnd_VeryHighEps_dqn_model.zip

### Experiment 19: Exp9_HighLR_MidGamma_SlowDecay
- **Member Name**: Alliance
- **Learning Rate**: 0.0008
- **Gamma**: 0.96
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.999
- **Noted Behavior**: Reward stable (-20.75 → -20.72), episode length decreased
- **Model Path**: models/Exp9_HighLR_MidGamma_SlowDecay_dqn_model.zip

### Experiment 20: Exp10_SmallBatch_HighGamma_LowEpsEnd
- **Member Name**: Alliance
- **Learning Rate**: 0.0003
- **Gamma**: 0.995
- **Batch Size**: 16
- **Epsilon Start**: 0.8
- **Epsilon End**: 0.005
- **Epsilon Decay**: 0.99
- **Noted Behavior**: Reward stable (-20.75 → -20.55), episode length increased
- **Model Path**: models/Exp10_SmallBatch_HighGamma_LowEpsEnd_dqn_model.zip

### Experiment 21: Stecie_Exp1_LowLR_LowGamma
- **Member Name**: Stecie
- **Learning Rate**: 1e-05
- **Gamma**: 0.9
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.59->-20.72), stable, episode length steady
- **Model Path**: models/Stecie_Exp1_LowLR_LowGamma_dqn_model.zip

### Experiment 22: Stecie_Exp2_HighLR_VeryHighGamma
- **Member Name**: Stecie
- **Learning Rate**: 0.0005
- **Gamma**: 0.999
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.93->-20.73), stable, episode length increased
- **Model Path**: models/Stecie_Exp2_HighLR_VeryHighGamma_dqn_model.zip

### Experiment 23: Stecie_Exp3_UltraLargeBatch
- **Member Name**: Stecie
- **Learning Rate**: 0.0002
- **Gamma**: 0.99
- **Batch Size**: 256
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.51->-20.80), stable, episode length steady
- **Model Path**: models/Stecie_Exp3_UltraLargeBatch_dqn_model.zip

### Experiment 24: Stecie_Exp4_UltraSmallBatch
- **Member Name**: Stecie
- **Learning Rate**: 0.0002
- **Gamma**: 0.99
- **Batch Size**: 4
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.38->-20.76), stable, episode length steady
- **Model Path**: models/Stecie_Exp4_UltraSmallBatch_dqn_model.zip

### Experiment 25: Stecie_Exp5_NoDecay_Epsilon
- **Member Name**: Stecie
- **Learning Rate**: 0.0002
- **Gamma**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.5
- **Epsilon Decay**: 1.0
- **Noted Behavior**: Reward mostly stable (-20.52->-20.53), stable, episode length increased
- **Model Path**: models/Stecie_Exp5_NoDecay_Epsilon_dqn_model.zip

### Experiment 26: Stecie_Exp6_FastDecay_Short
- **Member Name**: Stecie
- **Learning Rate**: 0.0002
- **Gamma**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.01
- **Epsilon Decay**: 0.95
- **Noted Behavior**: Reward mostly stable (-20.56->-20.67), stable, episode length increased
- **Model Path**: models/Stecie_Exp6_FastDecay_Short_dqn_model.zip

### Experiment 27: Stecie_Exp7_HighLR_LowGamma_SmallBatch
- **Member Name**: Stecie
- **Learning Rate**: 0.0008
- **Gamma**: 0.92
- **Batch Size**: 8
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.92->-20.65), stable, episode length increased
- **Model Path**: models/Stecie_Exp7_HighLR_LowGamma_SmallBatch_dqn_model.zip

### Experiment 28: Stecie_Exp8_LowLR_HighGamma_LargeBatch
- **Member Name**: Stecie
- **Learning Rate**: 1e-05
- **Gamma**: 0.995
- **Batch Size**: 128
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward mostly stable (-20.75->-20.58), stable, episode length increased
- **Model Path**: models/Stecie_Exp8_LowLR_HighGamma_LargeBatch_dqn_model.zip

### Experiment 29: Stecie_Exp9_ExtremeExploration
- **Member Name**: Stecie
- **Learning Rate**: 0.0002
- **Gamma**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.2
- **Epsilon Decay**: 0.98
- **Noted Behavior**: Reward mostly stable (-20.86->-20.65), stable, episode length decreased
- **Model Path**: models/Stecie_Exp9_ExtremeExploration_dqn_model.zip

### Experiment 30: Stecie_Exp10_MidRange_All
- **Member Name**: Stecie
- **Learning Rate**: 0.0003
- **Gamma**: 0.97
- **Batch Size**: 64
- **Epsilon Start**: 0.8
- **Epsilon End**: 0.03
- **Epsilon Decay**: 0.99
- **Noted Behavior**: Reward mostly stable (-20.92->-20.71), stable, episode length increased
- **Model Path**: models/Stecie_Exp10_MidRange_All_dqn_model.zip

### Experiment 31: 1M_Exp1_Balanced
- **Member Name**: Alliance
- **Learning Rate**: 0.0001
- **Gamma**: 0.99
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward improving (-20.50 -> -10.53), episode length increased
- **Model Path**: models/1M_Exp1_Balanced_dqn_model.zip

### Experiment 32: 1M_Exp2_LowGamma
- **Member Name**: Alliance
- **Learning Rate**: 0.0001
- **Gamma**: 0.8
- **Batch Size**: 32
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward improving (-20.50 -> -13.37), episode length increased
- **Model Path**: models/1M_Exp2_LowGamma_dqn_model.zip

### Experiment 33: 1M_Exp3_XLBatch
- **Member Name**: Alliance
- **Learning Rate**: 0.0001
- **Gamma**: 0.99
- **Batch Size**: 128
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: 0.995
- **Noted Behavior**: Reward improving (-20.50 -> -10.03), episode length increased
- **Model Path**: models/1M_Exp3_XLBatch_dqn_model.zip

