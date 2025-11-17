"""
Experiment Configuration for FL-NIDS
====================================
Defines all experiment phases and configurations.

Change PHASE to run different experiment sets:
- "baseline"    : No DP, no attacks, test all aggregators
- "privacy"     : Vary epsilon values, FedAvg only
- "aggregators" : Test robustness of different aggregators
- "attacks"     : Full attack scenarios
- "full"        : Complete grid (for final paper)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


class ExperimentConfig:
    """Centralized configuration with phase-based execution"""

    # ========================================================================
    # PHASE SELECTION - CHANGE THIS TO RUN DIFFERENT EXPERIMENTS
    # ========================================================================
    PHASE = "baseline"  # Options: baseline, privacy, aggregators, attacks, full

    # ========================================================================
    # PATHS
    # ========================================================================
    DATA_PATH = '/content/drive/MyDrive/IDSDatasets/UNSW 15'
    RESULTS_DIR = './results'
    LOGS_DIR = './logs'

    # ========================================================================
    # CORE PARAMETERS (Fixed across all experiments)
    # ========================================================================
    RANDOM_SEED = 42
    NUM_RUNS = 3  # Number of independent runs per config (reduced for faster execution)

    # Dataset
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1

    # Federated Learning
    NUM_CLIENTS = 10
    CLIENT_FRACTION = 1.0
    MIN_FIT_CLIENTS = 8
    MIN_AVAILABLE_CLIENTS = 8
    NUM_ROUNDS = 50  # Max rounds (can early stop, reduced from 100)

    # Training - Conservative approach for stable learning
    LOCAL_EPOCHS = 5  # Moderate local training
    BATCH_SIZE = 256  # Larger batches for stability
    LEARNING_RATE = 0.002  # Conservative learning rate

    # Model - Simpler architecture
    HIDDEN_DIMS = [64, 32]  # Smaller, simpler model
    DROPOUT_RATE = 0.1  # Minimal dropout

    # Differential Privacy
    TARGET_DELTA = 1e-5
    MAX_GRAD_NORM = 10.0  # Increased for DP stability (was 1.0, too strict)

    # Robust Aggregation
    TRIM_RATIO = 0.1

    # Early Stopping
    EARLY_STOP_PATIENCE = 10
    CONVERGENCE_THRESHOLD = 0.90  # F1 score

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========================================================================
    # PHASE CONFIGURATIONS
    # ========================================================================

    @classmethod
    def get_phase_config(cls) -> Dict[str, Any]:
        """Get configuration for current phase"""

        if cls.PHASE == "baseline":
            return {
                'name': 'Baseline Study',
                'description': 'Establish baseline performance with no DP, no attacks',
                'configs': [
                    {
                        'epsilon': float('inf'),
                        'aggregator': 'fedavg',
                        'attack_type': 'none',
                        'attack_ratio': 0.0,
                    },
                    {
                        'epsilon': float('inf'),
                        'aggregator': 'trimmed_mean',
                        'attack_type': 'none',
                        'attack_ratio': 0.0,
                    },
                    {
                        'epsilon': float('inf'),
                        'aggregator': 'median',
                        'attack_type': 'none',
                        'attack_ratio': 0.0,
                    },
                ],
                'num_rounds': 50,
                'use_sample': False,
                'sample_fraction': 1.0,
            }

        elif cls.PHASE == "privacy":
            return {
                'name': 'Privacy Analysis',
                'description': 'Analyze privacy-utility tradeoff across epsilon values',
                'configs': [
                    {'epsilon': eps, 'aggregator': 'fedavg', 'attack_type': 'none', 'attack_ratio': 0.0}
                    for eps in [0.5, 1.0, 5.0, float('inf')]
                ],
                'num_rounds': 50,
                'use_sample': False,
                'sample_fraction': 1.0,
            }

        elif cls.PHASE == "aggregators":
            return {
                'name': 'Aggregator Comparison',
                'description': 'Compare aggregation strategies under attacks (including Krum and model poisoning)',
                'configs': [
                    {
                        'epsilon': float('inf'),
                        'aggregator': agg,
                        'attack_type': attack,
                        'attack_ratio': ratio,
                    }
                    for agg in ['fedavg', 'trimmed_mean', 'median', 'krum']
                    for attack in ['none', 'label_flip', 'model_poisoning']
                    for ratio in ([0.0] if attack == 'none' else [0.2, 0.4])
                ],
                'num_rounds': 50,
                'use_sample': False,
                'sample_fraction': 1.0,
            }

        elif cls.PHASE == "attacks":
            return {
                'name': 'Attack Scenarios',
                'description': 'Comprehensive attack analysis with DP (epsilon 0.3, 0.5) and robust aggregators',
                'configs': [
                    {
                        'epsilon': eps,
                        'aggregator': agg,
                        'attack_type': attack,
                        'attack_ratio': ratio,
                    }
                    for eps in [0.3, 0.5, float('inf')]
                    for agg in ['fedavg', 'trimmed_mean', 'median', 'krum']
                    for attack in ['none', 'label_flip', 'model_poisoning']
                    for ratio in ([0.0] if attack == 'none' else [0.2, 0.4])
                ],
                'num_rounds': 50,
                'use_sample': False,
                'sample_fraction': 1.0,
            }

        elif cls.PHASE == "full":
            return {
                'name': 'Full Experimental Grid',
                'description': 'Complete experiments for final paper',
                'configs': [
                    {
                        'epsilon': eps,
                        'aggregator': agg,
                        'attack_type': attack,
                        'attack_ratio': ratio,
                    }
                    for eps in [0.5, 1.0, 5.0, float('inf')]
                    for agg in ['fedavg', 'trimmed_mean', 'median']
                    for attack in ['none', 'label_flip']
                    for ratio in ([0.0] if attack == 'none' else [0.1, 0.2, 0.3])
                ],
                'num_rounds': 50,
                'use_sample': False,
                'sample_fraction': 1.0,
            }

        else:
            raise ValueError(f"Unknown phase: {cls.PHASE}")

    @classmethod
    def get_result_dir(cls) -> Path:
        """Get results directory for current phase"""
        phase_dirs = {
            'baseline': 'phase1_baseline',
            'privacy': 'phase2_privacy',
            'aggregators': 'phase3_aggregators',
            'attacks': 'phase4_attacks',
            'full': 'phase5_full',
        }
        result_dir = Path(cls.RESULTS_DIR) / phase_dirs.get(cls.PHASE, cls.PHASE)
        result_dir.mkdir(parents=True, exist_ok=True)

        # Also create analysis subdirectory
        (result_dir / 'analysis').mkdir(exist_ok=True)

        return result_dir

    @classmethod
    def setup_directories(cls):
        """Create all necessary directories"""
        Path(cls.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.LOGS_DIR).mkdir(parents=True, exist_ok=True)
        cls.get_result_dir()

    @classmethod
    def set_seed(cls, seed: int = None):
        """Set random seeds for reproducibility"""
        seed = seed or cls.RANDOM_SEED
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @classmethod
    def print_phase_info(cls):
        """Print current phase configuration"""
        cfg = cls.get_phase_config()
        print("\n" + "="*80)
        print(f"PHASE: {cfg['name']}")
        print("="*80)
        print(f"Description: {cfg['description']}")
        print(f"Total configs: {len(cfg['configs'])}")
        print(f"Runs per config: {cls.NUM_RUNS}")
        print(f"Total experiments: {len(cfg['configs']) * cls.NUM_RUNS}")
        print(f"Max rounds: {cfg['num_rounds']}")
        print(f"Convergence threshold: F1 >= {cls.CONVERGENCE_THRESHOLD}")
        print(f"Early stop patience: {cls.EARLY_STOP_PATIENCE} rounds")
        print(f"\nResults will be saved to: {cls.get_result_dir()}")
        print("="*80 + "\n")

    @classmethod
    def get_experiment_filename(cls, config: Dict, run_id: int) -> str:
        """Generate filename for experiment result"""
        eps_str = f"eps_{config['epsilon']}" if config['epsilon'] != float('inf') else "eps_inf"
        agg_str = config['aggregator']
        attack_str = f"{config['attack_type']}_r{config['attack_ratio']}" if config['attack_ratio'] > 0 else "clean"

        return f"{eps_str}_{agg_str}_{attack_str}_run{run_id}.json"

    @classmethod
    def get_summary_filename(cls) -> str:
        """Get filename for phase summary"""
        return f"summary_{cls.PHASE}.json"


# Singleton instance
config = ExperimentConfig()
