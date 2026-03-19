"""
Validation Script for Task 1 Setup
Checks all dependencies and environment setup before training
"""

import sys
import subprocess
from importlib.util import find_spec

def check_module(module_name, display_name=None):
    """Check if a module is installed."""
    display_name = display_name or module_name
    print(f"Checking {display_name}...", end=" ")
    
    if find_spec(module_name) is not None:
        print("[OK]")
        return True
    else:
        print("[NOT FOUND]")
        return False

def check_version(module_name, min_version=None):
    """Check module version."""
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        if min_version:
            print(f"  Version: {version} (required: {min_version}+)")
        else:
            print(f"  Version: {version}")
        return True
    except:
        return False

def test_gymnasium():
    """Test Gymnasium environment creation."""
    print("\nTesting Gymnasium Environment...")
    try:
        import gymnasium as gym
        env = gym.make('ALE/Pong-v5', render_mode=None)
        print(f"  [OK] Pong environment created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        env.close()
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to create environment: {e}")
        return False

def test_stable_baselines():
    """Test Stable Baselines3 DQN."""
    print("\nTesting Stable Baselines3...")
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.dqn.policies import MlpPolicy, CnnPolicy
        print(f"  [OK] DQN imported successfully")
        print(f"  [OK] MlpPolicy imported successfully")
        print(f"  [OK] CnnPolicy imported successfully")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to import DQN: {e}")
        return False

def test_torch():
    """Test PyTorch installation and GPU availability."""
    print("\nTesting PyTorch...")
    try:
        import torch
        print(f"  [OK] PyTorch {torch.__version__} installed")
        if torch.cuda.is_available():
            print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print(f"  [WARNING] CUDA not available (will use CPU)")
            return True
    except Exception as e:
        print(f"  [ERROR] PyTorch error: {e}")
        return False

def main():
    """Run all validation checks."""
    print("="*70)
    print("TASK 1: Setup Validation Script")
    print("="*70)
    
    all_passed = True
    
    # Check Python version
    print(f"\nPython Version: {sys.version}")
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8+ required")
        all_passed = False
    else:
        print("[OK] Python version OK")
    
    # Check essential packages
    print("\n" + "-"*70)
    print("CHECKING DEPENDENCIES")
    print("-"*70)
    
    essential_packages = [
        ('stable_baselines3', 'Stable Baselines3'),
        ('gymnasium', 'Gymnasium'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('torch', 'PyTorch'),
    ]
    
    for package, display_name in essential_packages:
        if not check_module(package, display_name):
            all_passed = False
        else:
            check_version(package)
    
    # Test environment creation
    print("\n" + "-"*70)
    print("TESTING ENVIRONMENTS")
    print("-"*70)
    if not test_gymnasium():
        all_passed = False
    
    # Test Stable Baselines3
    print("\n" + "-"*70)
    print("TESTING STABLE BASELINES3")
    print("-"*70)
    if not test_stable_baselines():
        all_passed = False
    
    # Test PyTorch
    print("\n" + "-"*70)
    print("TESTING PYTORCH")
    print("-"*70)
    if not test_torch():
        all_passed = False
    
    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("[OK] ALL CHECKS PASSED - Ready to train!")
        print("\nTo start training, run:")
        print("  python train.py --mode final")
        print("\nOr run experiments:")
        print("  python train.py --mode experiments")
    else:
        print("[ERROR] SOME CHECKS FAILED - Please install missing dependencies")
        print("\nTo install all dependencies, run:")
        print("  pip install -r requirements.txt")
    print("="*70)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
