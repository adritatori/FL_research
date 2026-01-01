"""
Fairness Validation Experiment for Federated Learning with Differential Privacy
================================================================================

This script validates whether differential privacy disproportionately affects
detection of certain attack types in the UNSW-NB15 dataset.

Experimental Setup:
- Privacy levels: ε = ∞ (no DP), 5.0, 3.0, 1.0
- Aggregator: Median only
- Attack scenario: Clean (no Byzantine attacks)
- Runs per config: 3 (seeds: 42, 142, 242)
- Total: 12 experiments

Output: Per-attack-type detection rates and fairness metrics
"""

import os
import json
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

ATTACK_CATEGORIES = {
    'Normal': 0, 'Generic': 1, 'Exploits': 2, 'Fuzzers': 3,
    'DoS': 4, 'Reconnaissance': 5, 'Analysis': 6, 'Backdoor': 7,
    'Shellcode': 8, 'Worms': 9
}

ATTACK_NAMES = {v: k for k, v in ATTACK_CATEGORIES.items()}

# Experiment configurations
EPSILON_VALUES = [float('inf'), 5.0, 3.0, 1.0]
SEEDS = [42, 142, 242]
AGGREGATOR = "median"

# Federated Learning parameters
NUM_CLIENTS = 10
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
BATCH_SIZE = 256
BASE_LEARNING_RATE = 0.002
CLIENT_FRACTION = 1.0
MIN_FIT_CLIENTS = 8

# Differential Privacy parameters
TARGET_DELTA = 1e-5
MAX_GRAD_NORM = 10.0

# Adaptive learning rates for DP
def get_adaptive_lr(epsilon: float) -> float:
    """Get adaptive learning rate based on privacy budget."""
    if epsilon <= 1.0:
        return 0.05
    elif epsilon <= 5.0:
        return 0.01
    else:
        return 0.005

# Paths
RESULTS_DIR = Path("fairness_experiment/results")
MODELS_DIR = RESULTS_DIR / "models"
DATA_DIR = Path("/content/drive/MyDrive/IDSDatasets/UNSW 15")  # Google Drive path

# ============================================================================
# MODEL ARCHITECTURE (MUST MATCH PHASE 4)
# ============================================================================

class IntrusionDetectionMLP(nn.Module):
    """
    MLP for intrusion detection with GroupNorm (Opacus-compatible).

    Architecture:
    - Input → Linear(input_dim, 64) → GroupNorm(1, 64) → ReLU → Dropout(0.1)
    - → Linear(64, 32) → GroupNorm(1, 32) → ReLU → Dropout(0.1)
    - → Linear(32, 1) → Output (binary)
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.gn1 = nn.GroupNorm(1, 64)
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(64, 32)
        self.gn2 = nn.GroupNorm(1, 32)
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.gn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_unsw_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load UNSW-NB15 dataset preserving attack_cat column."""
    print(f"Loading data from {train_path} and {test_path}...")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    print(f"Features: {train_df.shape[1]}")

    return train_df, test_df

def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple:
    """
    Preprocess UNSW-NB15 data.

    Returns:
        X_train, y_train, attack_cat_train, X_test, y_test, attack_cat_test
    """
    # Identify feature columns (exclude label and attack_cat)
    exclude_cols = ['label', 'attack_cat']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    # Separate features and labels
    train_features = train_df[feature_cols].copy()
    test_features = test_df[feature_cols].copy()

    y_train = train_df['label'].values
    attack_cat_train = train_df['attack_cat'].map(ATTACK_CATEGORIES).values
    y_test = test_df['label'].values
    attack_cat_test = test_df['attack_cat'].map(ATTACK_CATEGORIES).values

    # Identify categorical columns (object/string dtype)
    categorical_cols = train_features.select_dtypes(include=['object']).columns.tolist()

    if categorical_cols:
        print(f"Encoding categorical columns: {categorical_cols}")
        # Apply one-hot encoding to categorical columns
        # Use pd.get_dummies which handles train/test consistency
        train_features = pd.get_dummies(train_features, columns=categorical_cols, drop_first=True)
        test_features = pd.get_dummies(test_features, columns=categorical_cols, drop_first=True)

        # Ensure train and test have the same columns (in case of missing categories)
        # Add missing columns to test set
        missing_cols = set(train_features.columns) - set(test_features.columns)
        for col in missing_cols:
            test_features[col] = 0

        # Add missing columns to train set (rare case)
        missing_cols = set(test_features.columns) - set(train_features.columns)
        for col in missing_cols:
            train_features[col] = 0

        # Ensure column order matches
        test_features = test_features[train_features.columns]

    # Convert to numpy arrays
    X_train = train_features.values
    X_test = test_features.values

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Attack category distribution (train):")
    unique, counts = np.unique(attack_cat_train, return_counts=True)
    for cat_id, count in zip(unique, counts):
        print(f"  {ATTACK_NAMES[cat_id]}: {count}")

    return X_train, y_train, attack_cat_train, X_test, y_test, attack_cat_test

def create_federated_data(X: np.ndarray, y: np.ndarray, attack_cat: np.ndarray,
                         num_clients: int, seed: int) -> List[Tuple]:
    """
    Partition data across clients using random split.

    Returns:
        List of (X_client, y_client, attack_cat_client) tuples
    """
    # Create dataset
    dataset = list(zip(X, y, attack_cat))

    # Shuffle with seed
    random.seed(seed)
    random.shuffle(dataset)

    # Split into equal parts
    client_size = len(dataset) // num_clients
    client_datasets = []

    for i in range(num_clients):
        start_idx = i * client_size
        end_idx = start_idx + client_size if i < num_clients - 1 else len(dataset)

        client_data = dataset[start_idx:end_idx]
        X_client = np.array([x[0] for x in client_data])
        y_client = np.array([x[1] for x in client_data])
        attack_cat_client = np.array([x[2] for x in client_data])

        client_datasets.append((X_client, y_client, attack_cat_client))
        print(f"Client {i}: {len(X_client)} samples")

    return client_datasets

# ============================================================================
# FEDERATED LEARNING CLIENT
# ============================================================================

class FLClient(fl.client.NumPyClient):
    """Flower client for federated learning with optional DP."""

    def __init__(self, model: nn.Module, X: np.ndarray, y: np.ndarray,
                 epsilon: float, delta: float, learning_rate: float,
                 local_epochs: int, batch_size: int, device: str):
        self.model = model.to(device)
        self.device = device
        self.epsilon = epsilon
        self.delta = delta
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size

        # Create dataset
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        self.dataset = TensorDataset(X_tensor, y_tensor)
        self.trainloader = DataLoader(self.dataset, batch_size=batch_size,
                                     shuffle=True, drop_last=True)

        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

        # Setup privacy engine if needed
        self.privacy_engine = None
        if not np.isinf(epsilon):
            # Make model compatible with Opacus
            self.model = ModuleValidator.fix(self.model)

            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.trainloader,
                noise_multiplier=1.0,  # Will be calibrated
                max_grad_norm=MAX_GRAD_NORM,
            )

    def get_parameters(self, config):
        """Return model parameters."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Set model parameters."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train model on local data."""
        self.set_parameters(parameters)
        self.model.train()

        for epoch in range(self.local_epochs):
            for batch_X, batch_y in self.trainloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

        # Get epsilon spent if using DP
        epsilon_spent = float('inf')
        if self.privacy_engine is not None:
            epsilon_spent = self.privacy_engine.get_epsilon(self.delta)

        return self.get_parameters(config), len(self.dataset), {
            "epsilon_spent": epsilon_spent
        }

    def evaluate(self, parameters, config):
        """Evaluate model on local data."""
        self.set_parameters(parameters)
        self.model.eval()

        loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in self.trainloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss += self.criterion(outputs, batch_y).item()

                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = loss / len(self.trainloader)

        return avg_loss, total, {"accuracy": accuracy}

# ============================================================================
# MEDIAN AGGREGATION STRATEGY
# ============================================================================

class MedianAggregation(Strategy):
    """Coordinate-wise median aggregation strategy."""

    def __init__(self, initial_parameters: Parameters, min_fit_clients: int,
                 min_available_clients: int):
        self.initial_parameters = initial_parameters
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients

    def initialize_parameters(self, client_manager):
        """Initialize global model parameters."""
        return self.initial_parameters

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Configure the next round of training."""
        # Sample all available clients
        sample_size = max(self.min_fit_clients,
                         int(client_manager.num_available() * CLIENT_FRACTION))
        clients = client_manager.sample(num_clients=sample_size)

        # Create fit config
        config = {"server_round": server_round}

        # Return client/config pairs
        return [(client, config) for client in clients]

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
                     failures):
        """Aggregate model updates using coordinate-wise median."""
        if not results:
            return None, {}

        # Extract parameters from results
        weights_list = [parameters_to_ndarrays(fit_res.parameters)
                       for _, fit_res in results]

        # Compute coordinate-wise median
        aggregated_weights = []
        for layer_idx in range(len(weights_list[0])):
            layer_weights = np.array([weights[layer_idx] for weights in weights_list])
            median_weight = np.median(layer_weights, axis=0)
            aggregated_weights.append(median_weight)

        # Collect epsilon spent
        epsilon_values = [fit_res.metrics.get("epsilon_spent", float('inf'))
                         for _, fit_res in results]
        max_epsilon = max(epsilon_values) if epsilon_values else float('inf')

        return ndarrays_to_parameters(aggregated_weights), {
            "epsilon_spent": max_epsilon
        }

    def configure_evaluate(self, server_round: int, parameters: Parameters,
                          client_manager):
        """Configure the next round of evaluation."""
        return []  # We'll evaluate on server side

    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation results."""
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters):
        """Evaluate global model (server-side)."""
        return None  # We'll handle evaluation separately

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray,
                  attack_cat: np.ndarray, device: str) -> Dict:
    """
    Evaluate model and compute per-attack-type detection rates.

    Returns:
        Dictionary with overall metrics and per-attack detection rates
    """
    model.eval()

    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy().flatten()

    # Overall metrics
    overall_metrics = {
        "f1": float(f1_score(y, predictions)),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0))
    }

    # Per-attack-type detection rates
    per_attack_detection = {}

    for cat_id in range(10):  # 0-9
        cat_name = ATTACK_NAMES[cat_id]
        cat_mask = (attack_cat == cat_id)

        if not cat_mask.any():
            continue

        cat_y = y[cat_mask]
        cat_pred = predictions[cat_mask]

        n_samples = int(cat_mask.sum())

        if cat_id == 0:  # Normal
            # True negative rate for normal traffic
            n_detected = int((cat_pred == 0).sum())
            detection_rate = float(n_detected / n_samples) if n_samples > 0 else 0.0
        else:  # Attack types
            # Detection rate for attacks (predicted as 1)
            n_detected = int((cat_pred == 1).sum())
            detection_rate = float(n_detected / n_samples) if n_samples > 0 else 0.0

        per_attack_detection[cat_name] = {
            "detection_rate": detection_rate,
            "n_samples": n_samples,
            "n_detected": n_detected
        }

    # Compute fairness metrics (attack types only, exclude Normal)
    attack_detection_rates = [
        metrics["detection_rate"]
        for cat_name, metrics in per_attack_detection.items()
        if cat_name != "Normal"
    ]

    if attack_detection_rates:
        min_detection = float(min(attack_detection_rates))
        max_detection = float(max(attack_detection_rates))
        gap = float(max_detection - min_detection)
        disparate_impact = float(min_detection / max_detection) if max_detection > 0 else 0.0
    else:
        min_detection = max_detection = gap = disparate_impact = 0.0

    fairness_metrics = {
        "min_detection": min_detection,
        "max_detection": max_detection,
        "gap": gap,
        "disparate_impact": disparate_impact
    }

    return {
        "overall_metrics": overall_metrics,
        "per_attack_detection": per_attack_detection,
        "fairness_metrics": fairness_metrics
    }

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(epsilon: float, seed: int, run_id: int,
                         X_train: np.ndarray, y_train: np.ndarray,
                         attack_cat_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         attack_cat_test: np.ndarray,
                         input_dim: int) -> Dict:
    """
    Run a single federated learning experiment.

    Returns:
        Results dictionary with config, metrics, and fairness analysis
    """
    print(f"\n{'='*80}")
    print(f"Running experiment: ε={epsilon}, seed={seed}, run_id={run_id}")
    print(f"{'='*80}\n")

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Get adaptive learning rate
    if np.isinf(epsilon):
        learning_rate = BASE_LEARNING_RATE
    else:
        learning_rate = get_adaptive_lr(epsilon)

    print(f"Learning rate: {learning_rate}")

    # Create federated data
    client_datasets = create_federated_data(X_train, y_train, attack_cat_train,
                                           NUM_CLIENTS, seed)

    # Create initial model
    initial_model = IntrusionDetectionMLP(input_dim)
    initial_params = [val.cpu().numpy() for val in initial_model.state_dict().values()]

    # Create strategy
    strategy = MedianAggregation(
        initial_parameters=ndarrays_to_parameters(initial_params),
        min_fit_clients=MIN_FIT_CLIENTS,
        min_available_clients=NUM_CLIENTS
    )

    # Create client function
    def client_fn(cid: str) -> FLClient:
        client_id = int(cid)
        X_client, y_client, _ = client_datasets[client_id]

        model = IntrusionDetectionMLP(input_dim)

        return FLClient(
            model=model,
            X=X_client,
            y=y_client,
            epsilon=epsilon,
            delta=TARGET_DELTA,
            learning_rate=learning_rate,
            local_epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            device=device
        )

    # Start simulation
    print(f"\nStarting federated learning simulation...")
    print(f"Clients: {NUM_CLIENTS}, Rounds: {NUM_ROUNDS}, Local epochs: {LOCAL_EPOCHS}")

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 0.4, "num_gpus": 0.09 if device == "cuda" else 0.0}  # A100: allows 10+ clients with plenty of VRAM headroom
    )

    # Get final model parameters
    final_params = parameters_to_ndarrays(history.parameters)

    # Create final model and load parameters
    final_model = IntrusionDetectionMLP(input_dim).to(device)
    params_dict = zip(final_model.state_dict().keys(), final_params)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    final_model.load_state_dict(state_dict)

    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    results = evaluate_model(final_model, X_test, y_test, attack_cat_test, device)

    # Add configuration
    results["config"] = {
        "epsilon": float(epsilon),
        "aggregator": AGGREGATOR,
        "seed": seed,
        "run_id": run_id,
        "learning_rate": learning_rate,
        "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE
    }

    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"{'='*80}")
    print(f"\nOverall Metrics:")
    for metric, value in results["overall_metrics"].items():
        print(f"  {metric}: {value:.4f}")

    print(f"\nPer-Attack Detection Rates:")
    for cat_name, metrics in sorted(results["per_attack_detection"].items()):
        print(f"  {cat_name:15s}: {metrics['detection_rate']:.4f} "
              f"({metrics['n_detected']}/{metrics['n_samples']})")

    print(f"\nFairness Metrics:")
    for metric, value in results["fairness_metrics"].items():
        print(f"  {metric}: {value:.4f}")

    # Save results
    eps_str = "inf" if np.isinf(epsilon) else f"{epsilon}"
    result_filename = RESULTS_DIR / f"eps_{eps_str}_run{run_id}.json"

    with open(result_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_filename}")

    # Save model
    model_filename = MODELS_DIR / f"eps_{eps_str}_run{run_id}.pt"
    torch.save(final_model.state_dict(), model_filename)
    print(f"Model saved to: {model_filename}")

    return results

def run_all_experiments():
    """Run all fairness validation experiments."""
    print("\n" + "="*80)
    print("FAIRNESS VALIDATION EXPERIMENT")
    print("="*80)
    print(f"\nConfigurations:")
    print(f"  Privacy levels: {EPSILON_VALUES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total experiments: {len(EPSILON_VALUES) * len(SEEDS)}")

    # Load and preprocess data
    train_path = DATA_DIR / "UNSW_NB15_training-set.csv"
    test_path = DATA_DIR / "UNSW_NB15_testing-set.csv"

    train_df, test_df = load_unsw_data(str(train_path), str(test_path))
    X_train, y_train, attack_cat_train, X_test, y_test, attack_cat_test = \
        preprocess_data(train_df, test_df)

    input_dim = X_train.shape[1]
    print(f"\nInput dimension: {input_dim}")

    # Run experiments
    all_results = []

    for epsilon in EPSILON_VALUES:
        for run_id, seed in enumerate(SEEDS):
            try:
                results = run_single_experiment(
                    epsilon=epsilon,
                    seed=seed,
                    run_id=run_id,
                    X_train=X_train,
                    y_train=y_train,
                    attack_cat_train=attack_cat_train,
                    X_test=X_test,
                    y_test=y_test,
                    attack_cat_test=attack_cat_test,
                    input_dim=input_dim
                )
                all_results.append(results)

            except Exception as e:
                print(f"\nERROR in experiment ε={epsilon}, seed={seed}: {str(e)}")
                import traceback
                traceback.print_exc()

    # Create summary CSV
    print(f"\n{'='*80}")
    print("Creating summary CSV...")
    print(f"{'='*80}\n")

    summary_rows = []
    for result in all_results:
        config = result["config"]
        overall = result["overall_metrics"]
        fairness = result["fairness_metrics"]

        row = {
            "epsilon": config["epsilon"],
            "seed": config["seed"],
            "run_id": config["run_id"],
            "f1": overall["f1"],
            "accuracy": overall["accuracy"],
            "precision": overall["precision"],
            "recall": overall["recall"],
            "min_detection": fairness["min_detection"],
            "max_detection": fairness["max_detection"],
            "detection_gap": fairness["gap"],
            "disparate_impact": fairness["disparate_impact"]
        }

        # Add per-attack detection rates
        for cat_name, metrics in result["per_attack_detection"].items():
            row[f"{cat_name}_detection"] = metrics["detection_rate"]

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    for epsilon in EPSILON_VALUES:
        eps_results = summary_df[summary_df["epsilon"] == epsilon]
        if len(eps_results) > 0:
            print(f"ε = {epsilon}:")
            print(f"  F1: {eps_results['f1'].mean():.4f} ± {eps_results['f1'].std():.4f}")
            print(f"  Detection gap: {eps_results['detection_gap'].mean():.4f} ± {eps_results['detection_gap'].std():.4f}")
            print(f"  Disparate impact: {eps_results['disparate_impact'].mean():.4f} ± {eps_results['disparate_impact'].std():.4f}")
            print()

    # Validation checks
    print(f"{'='*80}")
    print("VALIDATION CHECKS")
    print(f"{'='*80}\n")

    # Check 1: No DP achieves F1 > 0.93
    no_dp_f1 = summary_df[np.isinf(summary_df["epsilon"])]["f1"].values
    if len(no_dp_f1) > 0:
        max_no_dp_f1 = no_dp_f1.max()
        print(f"✓ No DP max F1: {max_no_dp_f1:.4f} (should be < 0.93)")

    # Check 2: ε=5.0 achieves F1 > 0.85
    eps5_f1 = summary_df[summary_df["epsilon"] == 5.0]["f1"].values
    if len(eps5_f1) > 0:
        mean_eps5_f1 = eps5_f1.mean()
        print(f"✓ ε=5.0 mean F1: {mean_eps5_f1:.4f} (should be > 0.85)")

    # Check 3: Variance check
    for epsilon in EPSILON_VALUES:
        eps_f1 = summary_df[summary_df["epsilon"] == epsilon]["f1"].values
        if len(eps_f1) > 0:
            std = eps_f1.std()
            print(f"✓ ε={epsilon} F1 std: {std:.4f} (should be < 0.1)")

    print(f"\n{'='*80}")
    print("EXPERIMENTS COMPLETED")
    print(f"{'='*80}\n")
    print(f"Total experiments run: {len(all_results)}/{len(EPSILON_VALUES) * len(SEEDS)}")
    print(f"Results directory: {RESULTS_DIR}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Run all experiments
    run_all_experiments()
