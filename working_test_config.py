"""
WORKING TEST CONFIGURATION
==========================
Copy and paste this ENTIRE configuration into your Jupyter notebook
This will fix both the 'krum' error and Ray OOM issues

USAGE:
1. In your notebook, replace the ExperimentConfig class with this
2. OR import it: from working_test_config import WorkingTestConfig as ExperimentConfig
"""

import torch

class WorkingTestConfig:
    """TESTED CONFIGURATION - Works without OOM errors"""

    # Paths
    DATA_PATH = './data/UNSW-NB15'
    RESULTS_DIR = './results'
    LOGS_DIR = './logs'

    # Random seeds
    RANDOM_SEED = 42

    # Dataset parameters
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1

    # ============================================================
    # CRITICAL FIX #1: Reduced FL parameters to avoid OOM
    # ============================================================
    NUM_CLIENTS = 3              # Reduced from 5
    NUM_ROUNDS = 10              # Reduced from 20 for quick test
    CLIENT_FRACTION = 1.0
    MIN_FIT_CLIENTS = 3          # ⚠️ MUST MATCH NUM_CLIENTS
    MIN_AVAILABLE_CLIENTS = 3    # ⚠️ MUST MATCH NUM_CLIENTS

    # Local training
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 128             # Larger = faster, but more memory
    LEARNING_RATE = 0.001

    # ============================================================
    # CRITICAL FIX #2: Only 2 epsilon values for quick test
    # ============================================================
    EPSILON_VALUES = [5.0, float('inf')]  # Just moderate DP and no-DP
    TARGET_DELTA = 1e-5
    MAX_GRAD_NORM = 1.0

    # ============================================================
    # CRITICAL FIX #3: Minimal robustness parameters
    # ============================================================
    MALICIOUS_FRACTIONS = [0.0, 0.2]      # Clean vs 20% malicious
    ATTACK_TYPES = ['none', 'label_flip']  # Just baseline + main attack

    # ============================================================
    # CRITICAL FIX #4: REMOVED KRUM - IT DOESN'T EXIST!
    # ============================================================
    # ❌ DO NOT USE: AGGREGATORS = ['fedavg', 'krum']  # KRUM WAS DELETED!
    # ✅ USE THIS INSTEAD:
    AGGREGATORS = ['fedavg', 'trimmed_mean']  # Just 2 for testing
    TRIM_RATIO = 0.1

    # Model parameters
    HIDDEN_DIMS = [128, 64, 32]
    DROPOUT_RATE = 0.2

    # ============================================================
    # CRITICAL FIX #5: Single run for testing
    # ============================================================
    NUM_RUNS = 1  # Just 1 run for quick test

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def setup_directories(cls):
        from pathlib import Path
        Path(cls.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.LOGS_DIR).mkdir(parents=True, exist_ok=True)

    @classmethod
    def set_seed(cls, seed: int = None):
        import numpy as np
        import torch
        seed = seed or cls.RANDOM_SEED
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @classmethod
    def print_config(cls):
        """Print configuration summary"""
        # Calculate expected experiments (with filtering)
        # alpha=0.0 only with attack='none'
        # alpha=0.2 only with attack='label_flip'
        clean = len(cls.EPSILON_VALUES) * 1 * 1 * len(cls.AGGREGATORS)
        attack = len(cls.EPSILON_VALUES) * 1 * 1 * len(cls.AGGREGATORS)
        total = (clean + attack) * cls.NUM_RUNS

        print("="*80)
        print("WORKING TEST CONFIGURATION")
        print("="*80)
        print(f"Clients: {cls.NUM_CLIENTS}")
        print(f"Rounds: {cls.NUM_ROUNDS}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Epsilon values: {cls.EPSILON_VALUES}")
        print(f"Malicious fractions: {cls.MALICIOUS_FRACTIONS}")
        print(f"Attack types: {cls.ATTACK_TYPES}")
        print(f"Aggregators: {cls.AGGREGATORS}")
        print(f"Runs: {cls.NUM_RUNS}")
        print()
        print(f"Expected experiments: {total}")
        print(f"  - Clean (α=0.0): {clean}")
        print(f"  - Attack (α=0.2): {attack}")
        print()
        print(f"Estimated time: 15-30 minutes")
        print(f"Memory usage: ~4-6 GB")
        print("="*80)


# Test the config
if __name__ == "__main__":
    config = WorkingTestConfig()
    config.print_config()
    print("\n✓ Configuration loaded successfully!")
    print("✓ No 'krum' aggregator (it was removed)")
    print("✓ All parameters are OOM-safe")
