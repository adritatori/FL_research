"""
Main Experiment Orchestrator
=============================
Phased execution with checkpoint/resume and progress tracking.

Usage:
    1. Set PHASE in config.py (baseline, privacy, aggregators, attacks, full)
    2. Run: python main.py
    3. Results saved incrementally
    4. Automatically resumes if interrupted

After experiments complete, runs statistical analysis.
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import modules
from config import config, ExperimentConfig
from metrics import MetricsTracker
from runner import run_single_experiment
from analysis import analyze_phase_results, create_summary_table, compare_aggregators


# ============================================================================
# DATA LOADING
# ============================================================================

class UNSWDataLoader:
    """Handle UNSW-NB15 dataset loading and preprocessing"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_and_preprocess(self, use_sample: bool = False,
                           sample_fraction: float = 1.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load UNSW-NB15 dataset with optional sampling"""
        print("=" * 80)
        print("LOADING UNSW-NB15 DATASET")
        print("=" * 80)

        # Try different file patterns
        possible_files = [
            'UNSW-NB15.csv',
            'UNSW_NB15_training-set.csv',
            'UNSW_NB15_testing-set.csv'
        ]

        df = None
        for filename in possible_files:
            filepath = os.path.join(self.data_path, filename)
            if os.path.exists(filepath):
                print(f"✓ Found: {filepath}")
                df = pd.read_csv(filepath, low_memory=False)
                break

        if df is None:
            # Try loading multiple files
            train_path = os.path.join(self.data_path, 'UNSW_NB15_training-set.csv')
            test_path = os.path.join(self.data_path, 'UNSW_NB15_testing-set.csv')

            if os.path.exists(train_path) and os.path.exists(test_path):
                print(f"✓ Loading training and testing sets separately")
                df_train = pd.read_csv(train_path, low_memory=False)
                df_test = pd.read_csv(test_path, low_memory=False)
                df = pd.concat([df_train, df_test], ignore_index=True)
            else:
                raise FileNotFoundError(f"Could not find UNSW-NB15 dataset in {self.data_path}")

        print(f"✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

        # Sample if requested
        if use_sample and sample_fraction < 1.0:
            original_size = len(df)
            df = df.sample(frac=sample_fraction, random_state=config.RANDOM_SEED)
            print(f"✓ Using {sample_fraction*100:.0f}% sample: {len(df):,} of {original_size:,} samples")

        # Handle label column
        label_col = 'label'
        print(f"✓ Using label column: '{label_col}'")

        y = df[label_col].values

        # Remove non-feature columns
        drop_cols = ['id', 'attack_cat', 'label']
        X = df.drop(columns=drop_cols, errors='ignore')

        # Handle categorical features
        categorical_cols = ['proto', 'service', 'state']
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        feature_names = list(X.columns)

        # Handle missing/infinite values
        X = X.apply(pd.to_numeric, errors='coerce')
        X = np.nan_to_num(X.values, nan=0.0, posinf=1e10, neginf=-1e10)

        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n✓ Class distribution:")
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count:,} ({count/len(y)*100:.2f}%)")

        print(f"\n✓ Final feature matrix: {X.shape}")
        print(f"✓ Label vector: {y.shape}")

        return X, y, feature_names


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def check_experiment_completed(results_dir: Path, exp_config: Dict, run_id: int) -> bool:
    """Check if experiment has already been completed

    Args:
        results_dir: Results directory
        exp_config: Experiment configuration
        run_id: Run ID

    Returns:
        True if completed
    """
    filename = config.get_experiment_filename(exp_config, run_id)
    filepath = results_dir / filename

    return filepath.exists()


def get_completed_experiments(results_dir: Path, configs: List[Dict]) -> int:
    """Count how many experiments have been completed

    Args:
        results_dir: Results directory
        configs: List of experiment configurations

    Returns:
        Number of completed experiments
    """
    completed = 0

    for exp_config in configs:
        for run_id in range(config.NUM_RUNS):
            if check_experiment_completed(results_dir, exp_config, run_id):
                completed += 1

    return completed


# ============================================================================
# MAIN EXPERIMENT EXECUTION
# ============================================================================

def run_phase_experiments():
    """Run all experiments for current phase with checkpoint/resume"""

    print("\n" + "="*80)
    print("FL-NIDS COMPREHENSIVE EXPERIMENT SYSTEM")
    print("="*80)

    # Setup
    config.setup_directories()
    config.print_phase_info()

    phase_cfg = config.get_phase_config()
    results_dir = config.get_result_dir()

    # Load data
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    data_loader = UNSWDataLoader(config.DATA_PATH)
    X, y, feature_names = data_loader.load_and_preprocess(
        use_sample=phase_cfg['use_sample'],
        sample_fraction=phase_cfg['sample_fraction']
    )

    # Split data
    print("\n" + "="*80)
    print("STEP 2: SPLITTING DATA")
    print("="*80)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")

    input_dim = X_train.shape[1]

    # Check for completed experiments
    total_experiments = len(phase_cfg['configs']) * config.NUM_RUNS
    completed = get_completed_experiments(results_dir, phase_cfg['configs'])

    print("\n" + "="*80)
    print("STEP 3: RUNNING EXPERIMENTS")
    print("="*80)
    print(f"Total experiments: {total_experiments}")
    print(f"Completed: {completed}")
    print(f"Remaining: {total_experiments - completed}")

    if completed == total_experiments:
        print("\n✓ All experiments already completed!")
        print("  Running analysis only...")
    elif completed > 0:
        print(f"\n↻ Resuming from checkpoint ({completed} experiments done)")

    print("\n" + "-"*80)

    # Run experiments
    experiment_num = 0
    start_time = time.time()

    for config_idx, exp_config in enumerate(phase_cfg['configs']):
        print(f"\n{'='*80}")
        print(f"Configuration {config_idx + 1}/{len(phase_cfg['configs'])}")
        print(f"{'='*80}")
        print(f"  Epsilon: {exp_config['epsilon']}")
        print(f"  Aggregator: {exp_config['aggregator']}")
        print(f"  Attack: {exp_config['attack_type']} ({exp_config['attack_ratio']*100:.0f}%)")
        print(f"{'='*80}")

        for run_id in range(config.NUM_RUNS):
            experiment_num += 1

            # Check if already completed
            if check_experiment_completed(results_dir, exp_config, run_id):
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Experiment {experiment_num}/{total_experiments}: SKIPPED (already completed)")
                continue

            print(f"\n{'='*80}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Experiment {experiment_num}/{total_experiments}")
            print(f"{'='*80}")

            # Run experiment
            try:
                tracker = run_single_experiment(
                    X_train, y_train, X_test, y_test,
                    input_dim,
                    exp_config,
                    run_id,
                    config
                )

                # Save results immediately
                filename = config.get_experiment_filename(exp_config, run_id)
                filepath = results_dir / filename
                tracker.save(filepath)

                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Saved: {filename}")

            except Exception as e:
                print(f"\n❌ ERROR in experiment {experiment_num}:")
                print(f"  {str(e)}")
                import traceback
                traceback.print_exc()
                print("\n  Continuing with next experiment...")
                continue

            # Progress update
            elapsed = time.time() - start_time
            completed_now = experiment_num
            avg_time = elapsed / completed_now
            remaining = total_experiments - completed_now
            est_remaining = avg_time * remaining

            print(f"\n{'='*80}")
            print(f"PROGRESS UPDATE")
            print(f"{'='*80}")
            print(f"  Completed: {completed_now}/{total_experiments} "
                  f"({completed_now/total_experiments*100:.1f}%)")
            print(f"  Elapsed time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
            print(f"  Avg time per experiment: {avg_time/60:.1f} minutes")
            print(f"  Est. remaining time: {est_remaining/3600:.2f} hours ({est_remaining/60:.1f} minutes)")
            print(f"{'='*80}\n")

    total_time = time.time() - start_time

    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"Average time per experiment: {total_time/total_experiments/60:.1f} minutes")
    print(f"Results saved to: {results_dir}")

    # Run analysis
    print("\n" + "="*80)
    print("STEP 4: STATISTICAL ANALYSIS")
    print("="*80)

    try:
        # Analyze key metrics
        for metric in ['final_f1', 'final_accuracy', 'best_f1']:
            print(f"\n{'='*80}")
            print(f"Analyzing metric: {metric}")
            print(f"{'='*80}")
            analyze_phase_results(results_dir, metric)

        # Create summary table
        print(f"\n{'='*80}")
        print("Creating summary table...")
        print(f"{'='*80}")
        create_summary_table(results_dir)

        # Compare aggregators (if applicable)
        if any('aggregator' in cfg for cfg in phase_cfg['configs']):
            print(f"\n{'='*80}")
            print("Comparing aggregators...")
            print(f"{'='*80}")
            compare_aggregators(results_dir)

    except Exception as e:
        print(f"\n⚠️  Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

    # Final instructions
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"✓ Results saved to: {results_dir}")
    print(f"✓ Analysis saved to: {results_dir / 'analysis'}")
    print(f"\nTo run another phase:")
    print(f"  1. Edit config.py and change PHASE")
    print(f"  2. Run: python main.py")
    print("\nAvailable phases: baseline, privacy, aggregators, attacks, full")
    print("="*80 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    try:
        run_phase_experiments()

    except FileNotFoundError as e:
        print(f"\n❌ CRITICAL ERROR: Could not find data file.")
        print(f"  Details: {e}")
        print(f"  Please update DATA_PATH in config.py (currently: {config.DATA_PATH})")
        sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n\n⚠️  INTERRUPTED BY USER")
        print(f"  Experiments can be resumed by running: python main.py")
        print(f"  Already completed experiments will be skipped.")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
