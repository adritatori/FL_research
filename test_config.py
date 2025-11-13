"""
Quick test configuration for NWDAF Security Analytics
Use this for rapid testing with minimal resource usage
"""

class TestConfig:
    """Minimal config for testing - completes in ~10-30 minutes"""

    # Paths
    DATA_PATH = './data/UNSW-NB15'
    RESULTS_DIR = './results'
    LOGS_DIR = './logs'

    # Random seeds
    RANDOM_SEED = 42

    # Dataset parameters
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1

    # MINIMAL FL PARAMETERS FOR TESTING
    NUM_CLIENTS = 3              # Reduced from 5
    NUM_ROUNDS = 10              # Reduced from 20
    CLIENT_FRACTION = 1.0
    MIN_FIT_CLIENTS = 3          # Match NUM_CLIENTS
    MIN_AVAILABLE_CLIENTS = 3    # Match NUM_CLIENTS

    # Local training
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 128             # Larger batch = faster
    LEARNING_RATE = 0.001

    # Differential Privacy - ONLY 2 VALUES FOR TESTING
    EPSILON_VALUES = [5.0, float('inf')]  # Just moderate DP and no-DP
    TARGET_DELTA = 1e-5
    MAX_GRAD_NORM = 1.0

    # Robustness - MINIMAL SET
    MALICIOUS_FRACTIONS = [0.0, 0.2]      # Clean vs 20% malicious
    ATTACK_TYPES = ['none', 'label_flip']  # Just baseline and main attack

    # ⚠️ IMPORTANT: Only use aggregators that exist!
    # Krum was removed - DO NOT ADD IT BACK
    AGGREGATORS = ['fedavg', 'trimmed_mean']  # Just 2 for testing
    TRIM_RATIO = 0.1

    # Model parameters
    HIDDEN_DIMS = [128, 64, 32]
    DROPOUT_RATE = 0.2

    # Experiment parameters
    NUM_RUNS = 1  # Single run for testing

    # Device
    import torch
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
    def print_summary(cls):
        """Print expected experiment count"""
        clean_configs = len(cls.EPSILON_VALUES) * 1 * 1 * len(cls.AGGREGATORS)
        attack_configs = len(cls.EPSILON_VALUES) * 1 * 1 * len(cls.AGGREGATORS)
        total_per_run = clean_configs + attack_configs
        total = total_per_run * cls.NUM_RUNS

        print("="*80)
        print("TEST CONFIGURATION SUMMARY")
        print("="*80)
        print(f"Clients: {cls.NUM_CLIENTS}")
        print(f"Rounds: {cls.NUM_ROUNDS}")
        print(f"Epsilon values: {cls.EPSILON_VALUES}")
        print(f"Malicious fractions: {cls.MALICIOUS_FRACTIONS}")
        print(f"Attack types: {cls.ATTACK_TYPES}")
        print(f"Aggregators: {cls.AGGREGATORS}")
        print(f"Runs: {cls.NUM_RUNS}")
        print()
        print(f"Expected experiments: {total}")
        print(f"Estimated time: 10-30 minutes")
        print("="*80)
