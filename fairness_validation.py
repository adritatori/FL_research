"""
Fairness Validation for FL-NIDS (Binary Classification) - FIXED
================================================================
Direct evaluation of final global model to collect predictions.

This version:
1. Trains the federated model normally
2. After training completes, evaluates the FINAL GLOBAL MODEL directly
3. Collects predictions from this evaluation
4. Performs per-attack-type fairness analysis

Key insight: Don't rely on Flower's metric aggregation for predictions.
Instead, evaluate the final model parameters directly.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import flwr as fl
from flwr.common import NDArrays, Parameters, Metrics
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from opacus import PrivacyEngine


# ============================================================================
# CONFIGURATION
# ============================================================================

class FairnessConfig:
    """Configuration - EXACT MATCH to Phase 4"""

    # Paths
    DATA_PATH = '/content/drive/MyDrive/IDSDatasets/UNSW 15'
    OUTPUT_DIR = './fairness_results'

    # Fixed parameters
    RANDOM_SEED = 42
    NUM_CLIENTS = 10
    CLIENT_FRACTION = 1.0
    MIN_FIT_CLIENTS = 8
    MIN_AVAILABLE_CLIENTS = 8
    NUM_ROUNDS = 50
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 256
    LEARNING_RATE = 0.002
    HIDDEN_DIMS = [64, 32]
    DROPOUT_RATE = 0.1
    TARGET_DELTA = 1e-5
    MAX_GRAD_NORM = 10.0
    TEST_SIZE = 0.2

    # Device configuration
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        DEVICE = torch.device('cpu')
        print("WARNING: CUDA not available, using CPU")

    # Experiments
    EXPERIMENTS = [
        {'epsilon': float('inf'), 'aggregator': 'median', 'attack_type': 'label_flip', 'attack_ratio': 0.4, 'name': 'No DP (Baseline)'},
        {'epsilon': 5.0, 'aggregator': 'median', 'attack_type': 'label_flip', 'attack_ratio': 0.4, 'name': 'ε=5.0 (Relaxed)'},
        {'epsilon': 3.0, 'aggregator': 'median', 'attack_type': 'label_flip', 'attack_ratio': 0.4, 'name': 'ε=3.0 (Safe Region)'},
        {'epsilon': 1.0, 'aggregator': 'median', 'attack_type': 'label_flip', 'attack_ratio': 0.4, 'name': 'ε=1.0 (Danger Zone)'},
    ]

    # Attack category mapping
    ATTACK_CATEGORIES = {
        'Normal': 0, 'Generic': 1, 'Exploits': 2, 'Fuzzers': 3, 'DoS': 4,
        'Reconnaissance': 5, 'Analysis': 6, 'Backdoor': 7, 'Shellcode': 8, 'Worms': 9
    }

    CATEGORY_NAMES = {v: k for k, v in ATTACK_CATEGORIES.items()}


config = FairnessConfig()


# ============================================================================
# NEURAL NETWORK
# ============================================================================

class IntrusionDetectionMLP(nn.Module):
    """Binary intrusion detection model"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GroupNorm(1, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================================
# FEDERATED LEARNING CLIENTS
# ============================================================================

class BaseClient(fl.client.NumPyClient):
    """Standard FL client"""

    def __init__(self, cid: int, model: nn.Module, trainloader: DataLoader, testloader: DataLoader):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

        # Calculate pos_weight
        all_labels = []
        for _, labels in trainloader:
            all_labels.extend(labels.numpy().flatten())
        all_labels = np.array(all_labels)

        num_class_0 = np.sum(all_labels == 0)
        num_class_1 = np.sum(all_labels == 1)

        if num_class_1 > 0:
            pos_weight = torch.tensor([num_class_0 / num_class_1]).to(config.DEVICE)
        else:
            pos_weight = torch.tensor([1.0]).to(config.DEVICE)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=1e-4
        )
        self.privacy_engine = None

    def setup_dp(self, epsilon: float, num_rounds: int):
        """Setup differential privacy"""
        if epsilon == float('inf'):
            return

        # Adaptive LR
        if epsilon <= 1.0:
            adaptive_lr = 0.05
        elif epsilon <= 5.0:
            adaptive_lr = 0.01
        else:
            adaptive_lr = 0.005

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=adaptive_lr,
            momentum=0.9,
            weight_decay=1e-4
        )

        self.privacy_engine = PrivacyEngine()

        self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.trainloader,
            target_epsilon=epsilon,
            target_delta=config.TARGET_DELTA,
            epochs=num_rounds * config.LOCAL_EPOCHS,
            max_grad_norm=config.MAX_GRAD_NORM,
        )

    def get_parameters(self, config_dict: Dict) -> NDArrays:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config_dict: Dict) -> Tuple[NDArrays, int, Dict]:
        self.set_parameters(parameters)
        self.model.train()

        total_loss = 0
        num_batches = 0

        for epoch in range(config.LOCAL_EPOCHS):
            for data, target in self.trainloader:
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                if torch.isnan(loss):
                    continue

                loss.backward()

                if self.privacy_engine is None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        metrics = {
            'client_id': self.cid,
            'avg_loss': avg_loss,
            'num_examples': len(self.trainloader.dataset),
        }

        if self.privacy_engine:
            epsilon = self.privacy_engine.get_epsilon(delta=config.TARGET_DELTA)
            metrics['epsilon_consumed'] = epsilon

        return self.get_parameters({}), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters: NDArrays, config_dict: Dict) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()

        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)

                logits = self.model(data)
                output = torch.sigmoid(logits)

                loss = self.criterion(logits, target)
                total_loss += loss.item()

                preds = (output > 0.5).float().cpu().numpy()
                all_preds.extend(preds.flatten())
                all_labels.extend(target.cpu().numpy().flatten())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        f1 = f1_score(all_labels, all_preds, zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds)

        metrics = {
            'loss': total_loss / len(self.testloader) if len(self.testloader) > 0 else 0,
            'num_examples': len(all_labels),
            'f1': float(f1),
            'accuracy': float(accuracy),
        }

        return float(metrics['loss']), len(self.testloader.dataset), metrics


class MaliciousClient(BaseClient):
    """Client performing label flip attack"""

    def fit(self, parameters: NDArrays, config_dict: Dict) -> Tuple[NDArrays, int, Dict]:
        self.set_parameters(parameters)
        self.model.train()

        for epoch in range(config.LOCAL_EPOCHS):
            for data, target in self.trainloader:
                flipped_target = 1 - target
                data = data.to(config.DEVICE)
                flipped_target = flipped_target.to(config.DEVICE)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, flipped_target)
                loss.backward()
                self.optimizer.step()

        metrics = {
            'client_id': self.cid,
            'attack_type': 'label_flip',
            'num_examples': len(self.trainloader.dataset),
        }

        return self.get_parameters({}), len(self.trainloader.dataset), metrics


# ============================================================================
# AGGREGATION STRATEGY
# ============================================================================

class MedianStrategy(FedAvg):
    """Coordinate-wise median aggregation"""

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        weights = [w for w, _ in weights_results]
        aggregated = []

        for layer_idx in range(len(weights[0])):
            layer_weights = np.stack([w[layer_idx] for w in weights])
            aggregated_layer = np.median(layer_weights, axis=0)
            aggregated.append(aggregated_layer)

        parameters_aggregated = ndarrays_to_parameters(aggregated)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated


# ============================================================================
# DIRECT MODEL EVALUATION (KEY FIX)
# ============================================================================

def evaluate_final_model(final_parameters: NDArrays, X_test: np.ndarray, y_test: np.ndarray,
                         input_dim: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Evaluate the final global model directly on test data.

    This is the KEY function that solves the prediction collection problem.
    """
    print("  [EVAL] Evaluating final global model...")

    # Create fresh model
    model = IntrusionDetectionMLP(
        input_dim=input_dim,
        hidden_dims=config.HIDDEN_DIMS,
        dropout=config.DROPOUT_RATE
    ).to(config.DEVICE)

    # Load final parameters
    params_dict = zip(model.state_dict().keys(), final_parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

    # Evaluate
    model.eval()

    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test).view(-1, 1)
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(config.DEVICE)

            logits = model(data)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(target.numpy().flatten())

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)

    # Compute metrics
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"  [EVAL] Final model F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
    print(f"  [EVAL] Predictions collected: {len(y_pred)}")

    return y_pred, y_prob, f1, accuracy


# ============================================================================
# DATA LOADING
# ============================================================================

def load_unsw_data():
    """Load UNSW-NB15 with attack categories preserved"""

    print("=" * 80)
    print("LOADING UNSW-NB15 DATASET")
    print("=" * 80)

    df = None
    for filename in ['UNSW-NB15.csv', 'UNSW_NB15_training-set.csv']:
        filepath = os.path.join(config.DATA_PATH, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, low_memory=False)
            print(f"✓ Loaded: {filepath}")
            break

    if df is None:
        train_path = os.path.join(config.DATA_PATH, 'UNSW_NB15_training-set.csv')
        test_path = os.path.join(config.DATA_PATH, 'UNSW_NB15_testing-set.csv')
        if os.path.exists(train_path) and os.path.exists(test_path):
            df = pd.concat([pd.read_csv(train_path, low_memory=False),
                           pd.read_csv(test_path, low_memory=False)], ignore_index=True)
            print(f"✓ Loaded train + test sets")

    if df is None:
        raise FileNotFoundError(f"Dataset not found in {config.DATA_PATH}")

    print(f"✓ Total samples: {len(df)}")

    # Extract attack categories
    if 'attack_cat' in df.columns:
        attack_cat_raw = df['attack_cat'].fillna('Normal').values
        attack_categories = np.array([config.ATTACK_CATEGORIES.get(str(cat).strip(), 0) for cat in attack_cat_raw])
    else:
        print("⚠ No attack_cat column - using binary labels only")
        attack_categories = df['label'].values

    # Print distribution
    print("\nAttack Category Distribution:")
    for cat_id in range(10):
        count = np.sum(attack_categories == cat_id)
        if count > 0:
            print(f"  {config.CATEGORY_NAMES[cat_id]:15s}: {count:6,} ({count/len(df)*100:5.2f}%)")

    # Binary labels
    y_binary = df['label'].values

    # Features
    drop_cols = ['id', 'attack_cat', 'label']
    X = df.drop(columns=drop_cols, errors='ignore')

    # Encode categoricals
    for col in ['proto', 'service', 'state']:
        if col in X.columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.apply(pd.to_numeric, errors='coerce')
    X = np.nan_to_num(X.values, nan=0.0, posinf=1e10, neginf=-1e10)

    return X, y_binary, attack_categories


def iid_partition(X: np.ndarray, y: np.ndarray, num_clients: int) -> List[Tuple]:
    """IID data partitioning"""
    indices = np.random.permutation(len(X))
    splits = np.array_split(indices, num_clients)
    return [(X[split], y[split]) for split in splits]


# ============================================================================
# SINGLE EXPERIMENT RUNNER (FIXED)
# ============================================================================

def run_experiment(X_train, y_train, X_test, y_test, attack_categories_test,
                   exp_config: Dict, input_dim: int) -> Dict:
    """Run single fairness experiment with DIRECT final model evaluation"""

    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_config['name']}")
    print(f"{'='*80}")

    # Set seed
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    # Partition data
    partitions = iid_partition(X_train, y_train, config.NUM_CLIENTS)

    # Malicious client IDs
    num_malicious = int(exp_config['attack_ratio'] * config.NUM_CLIENTS)
    malicious_ids = list(range(num_malicious))
    print(f"Malicious clients: {malicious_ids}")

    def client_fn(cid: str):
        client_id = int(cid)
        X_client, y_client = partitions[client_id]

        train_dataset = TensorDataset(
            torch.FloatTensor(X_client),
            torch.FloatTensor(y_client).view(-1, 1)
        )
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test).view(-1, 1)
        )
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

        model = IntrusionDetectionMLP(
            input_dim=input_dim,
            hidden_dims=config.HIDDEN_DIMS,
            dropout=config.DROPOUT_RATE
        ).to(config.DEVICE)

        if client_id in malicious_ids and exp_config['attack_type'] != 'none':
            return MaliciousClient(
                client_id, model, train_loader, test_loader
            ).to_client()
        else:
            client = BaseClient(client_id, model, train_loader, test_loader)
            if exp_config['epsilon'] != float('inf'):
                client.setup_dp(exp_config['epsilon'], config.NUM_ROUNDS)
            return client.to_client()

    # Metrics aggregation
    def weighted_avg(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        if not metrics:
            return {}

        aggregated = {}

        for key in ['accuracy', 'f1', 'loss']:
            if all(key in m for _, m in metrics):
                values = [m[key] * n for n, m in metrics]
                total = sum(n for n, _ in metrics)
                if total > 0:
                    aggregated[key] = sum(values) / total

        epsilon_values = [m.get('epsilon_consumed', 0.0) for _, m in metrics if 'epsilon_consumed' in m]
        if epsilon_values:
            aggregated['epsilon_consumed'] = max(epsilon_values)

        return aggregated

    # Initialize global model
    print(f"Initializing global model...")
    init_model = IntrusionDetectionMLP(
        input_dim=input_dim,
        hidden_dims=config.HIDDEN_DIMS,
        dropout=config.DROPOUT_RATE
    ).to(config.DEVICE)

    initial_parameters = ndarrays_to_parameters([
        val.cpu().numpy() for _, val in init_model.state_dict().items()
    ])

    # Strategy
    strategy_params = {
        "fraction_fit": config.CLIENT_FRACTION,
        "fraction_evaluate": 1.0,
        "min_fit_clients": config.MIN_FIT_CLIENTS,
        "min_available_clients": config.MIN_AVAILABLE_CLIENTS,
        "evaluate_metrics_aggregation_fn": weighted_avg,
        "fit_metrics_aggregation_fn": weighted_avg,
        "initial_parameters": initial_parameters,
    }

    strategy = MedianStrategy(**strategy_params)

    print(f"Starting FL training...")

    # Run simulation
    start_time = time.time()

    client_resources = {
        "num_cpus": 0.3,  # Reduced to allow 5+ clients (was 1, which only allowed 2 actors)
        "num_gpus": 0.15 if torch.cuda.is_available() else 0.0  # Adjusted for 5-6 actors
    }

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=config.NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources
    )

    total_time = time.time() - start_time

    print(f"\n✓ Training complete! Time: {total_time:.1f}s")

    # KEY FIX: Get final parameters and evaluate directly
    final_parameters = None
    if history.parameters:
        final_parameters = parameters_to_ndarrays(history.parameters)
        print(f"✓ Final parameters extracted from history")
    else:
        print("⚠ Warning: Could not extract final parameters from history")
        print("  Attempting to retrieve from strategy...")
        # Fallback: try to get from strategy
        if hasattr(strategy, 'current_parameters') and strategy.current_parameters:
            final_parameters = parameters_to_ndarrays(strategy.current_parameters)
            print(f"✓ Final parameters extracted from strategy")

    # Evaluate final model directly
    if final_parameters is not None:
        y_pred, y_prob, final_f1, final_acc = evaluate_final_model(
            final_parameters, X_test, y_test, input_dim
        )
    else:
        print("✗ ERROR: Could not retrieve final model parameters!")
        y_pred = np.array([])
        y_prob = np.array([])
        final_f1 = 0.0
        final_acc = 0.0

    return {
        'config': exp_config,
        'final_f1': final_f1,
        'final_accuracy': final_acc,
        'total_time': total_time,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'y_true': y_test,
        'attack_categories': attack_categories_test
    }


# ============================================================================
# FAIRNESS ANALYSIS (unchanged from original)
# ============================================================================

def compute_detection_rates(y_pred: np.ndarray, y_true: np.ndarray,
                            attack_categories: np.ndarray) -> Dict:
    """Compute detection rate per attack category"""

    if len(y_pred) == 0:
        return {}

    results = {}

    for cat_id, cat_name in config.CATEGORY_NAMES.items():
        mask = (attack_categories == cat_id)
        n_samples = mask.sum()

        if n_samples == 0:
            continue

        if cat_name == 'Normal':
            correct = (y_pred[mask] == 0).sum()
            detection_rate = correct / n_samples
            metric_name = 'True Negative Rate'
        else:
            correct = (y_pred[mask] == 1).sum()
            detection_rate = correct / n_samples
            metric_name = 'Detection Rate (Recall)'

        results[cat_name] = {
            'detection_rate': detection_rate,
            'n_samples': int(n_samples),
            'n_detected': int(correct),
            'metric': metric_name
        }

    return results


def analyze_fairness(results: List[Dict], output_dir: Path):
    """Analyze per-attack-type detection rates"""

    print(f"\n{'='*80}")
    print("FAIRNESS ANALYSIS: Per-Attack-Type Detection Rates")
    print(f"{'='*80}\n")

    all_detection_rates = []

    for result in results:
        exp_name = result['config']['name']
        epsilon = result['config']['epsilon']

        if 'y_pred' not in result or len(result['y_pred']) == 0:
            print(f"\n{exp_name} (ε={epsilon}):")
            print(f"  Overall F1: {result['final_f1']:.4f}")
            print(f"  ✗ Per-attack analysis unavailable")
            continue

        detection = compute_detection_rates(
            result['y_pred'],
            result['y_true'],
            result['attack_categories']
        )

        if not detection:
            continue

        print(f"\n{exp_name} (ε={epsilon}):")
        print(f"  Overall F1: {result['final_f1']:.4f}")
        print(f"  Per-Attack Detection Rates:")

        attack_rates = []
        for cat_name, data in detection.items():
            rate = data['detection_rate']
            n = data['n_samples']
            print(f"    {cat_name:15s}: {rate*100:5.1f}% ({data['n_detected']}/{n})")

            if cat_name != 'Normal':
                attack_rates.append(rate)

            all_detection_rates.append({
                'epsilon': epsilon,
                'epsilon_str': 'No DP' if epsilon == float('inf') else f'ε={epsilon}',
                'experiment': exp_name,
                'attack_type': cat_name,
                'detection_rate': rate,
                'n_samples': n,
                'overall_f1': result['final_f1']
            })

        # Fairness metrics
        if attack_rates:
            min_rate = min(attack_rates)
            max_rate = max(attack_rates)
            gap = max_rate - min_rate
            di = min_rate / max_rate if max_rate > 0 else 0

            print(f"\n  Fairness Metrics:")
            print(f"    Min Detection: {min_rate*100:.1f}%")
            print(f"    Max Detection: {max_rate*100:.1f}%")
            print(f"    Gap: {gap*100:.1f}%")
            print(f"    Disparate Impact: {di:.3f} {'✓' if di >= 0.8 else '✗'}")

    if not all_detection_rates:
        print("\n✗ ERROR: No detection rates computed!")
        return None

    df = pd.DataFrame(all_detection_rates)
    df.to_csv(output_dir / 'detection_rates.csv', index=False)
    print(f"\n✓ Saved: {output_dir / 'detection_rates.csv'}")

    return df


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create fairness visualizations"""

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    epsilon_order = ['No DP', 'ε=5.0', 'ε=3.0', 'ε=1.0']
    available_eps = [e for e in epsilon_order if e in df['epsilon_str'].values]

    attack_order = ['Generic', 'Exploits', 'Fuzzers', 'DoS', 'Reconnaissance',
                    'Analysis', 'Backdoor', 'Shellcode', 'Worms']

    # 1. Per-Attack Detection Rate Comparison
    ax = axes[0, 0]
    attacks_df = df[df['attack_type'] != 'Normal']

    x = np.arange(len(attack_order))
    width = 0.8 / len(available_eps)

    for i, eps_str in enumerate(available_eps):
        subset = attacks_df[attacks_df['epsilon_str'] == eps_str]
        rates = []
        for attack in attack_order:
            rate_data = subset[subset['attack_type'] == attack]['detection_rate']
            rates.append(rate_data.values[0] if len(rate_data) > 0 else 0)

        ax.bar(x + i * width, [r * 100 for r in rates], width, label=eps_str, alpha=0.8)

    ax.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Detection Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Attack-Type Detection Rates', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * len(available_eps) / 2)
    ax.set_xticklabels(attack_order, rotation=45, ha='right')
    ax.legend(title='Privacy Level')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Detection Rate Heatmap
    ax = axes[0, 1]

    pivot = attacks_df.pivot_table(
        values='detection_rate',
        index='attack_type',
        columns='epsilon_str',
        aggfunc='mean'
    )

    pivot = pivot.reindex(index=[a for a in attack_order if a in pivot.index])
    pivot = pivot[[c for c in available_eps if c in pivot.columns]]

    sns.heatmap(pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Detection Rate (%)'})
    ax.set_title('Detection Rate Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Privacy Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Type', fontsize=12, fontweight='bold')

    # 3. Rare vs Common Classes
    ax = axes[1, 0]

    common_attacks = ['Generic', 'Exploits', 'DoS']
    rare_attacks = ['Analysis', 'Backdoor', 'Shellcode', 'Worms']

    common_rates = []
    rare_rates = []

    for eps_str in available_eps:
        subset = attacks_df[attacks_df['epsilon_str'] == eps_str]

        common_data = subset[subset['attack_type'].isin(common_attacks)]['detection_rate']
        rare_data = subset[subset['attack_type'].isin(rare_attacks)]['detection_rate']

        common_rates.append(common_data.mean() * 100 if len(common_data) > 0 else 0)
        rare_rates.append(rare_data.mean() * 100 if len(rare_data) > 0 else 0)

    x = np.arange(len(available_eps))
    width = 0.35

    ax.bar(x - width/2, common_rates, width, label='Common Attacks', color='#2ca02c', alpha=0.8)
    ax.bar(x + width/2, rare_rates, width, label='Rare Attacks', color='#d62728', alpha=0.8)

    ax.set_xlabel('Privacy Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg Detection Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Common vs Rare Attack Detection', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(available_eps)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Fairness Disparity
    ax = axes[1, 1]

    disparities = []
    gaps = []

    for eps_str in available_eps:
        subset = attacks_df[attacks_df['epsilon_str'] == eps_str]
        rates = subset['detection_rate'].values

        if len(rates) > 0 and max(rates) > 0:
            di = min(rates) / max(rates)
            gap = (max(rates) - min(rates)) * 100
        else:
            di = 0
            gap = 0

        disparities.append(di)
        gaps.append(gap)

    ax2 = ax.twinx()

    line1 = ax.plot(available_eps, disparities, 'o-', color='#1f77b4', linewidth=2, markersize=10, label='Disparate Impact')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% Rule')

    line2 = ax2.plot(available_eps, gaps, 's-', color='#ff7f0e', linewidth=2, markersize=10, label='Performance Gap')

    ax.set_xlabel('Privacy Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Disparate Impact Ratio', fontsize=12, fontweight='bold', color='#1f77b4')
    ax2.set_ylabel('Performance Gap (%)', fontsize=12, fontweight='bold', color='#ff7f0e')
    ax.set_title('Fairness Metrics vs Privacy Level', fontsize=14, fontweight='bold')

    ax.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best')

    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fairness_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_dir / 'fairness_analysis.png'}")


def generate_report(df: pd.DataFrame, results: List[Dict], output_dir: Path):
    """Generate text report"""

    report_path = output_dir / 'fairness_report.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FAIRNESS VALIDATION REPORT\n")
        f.write("Per-Attack-Type Detection Rates in FL-NIDS\n")
        f.write("=" * 80 + "\n\n")

        attacks_df = df[df['attack_type'] != 'Normal']

        for result in results:
            eps = result['config']['epsilon']
            eps_str = 'No DP' if eps == float('inf') else f'ε={eps}'

            f.write(f"\n{'='*80}\n")
            f.write(f"EXPERIMENT: {eps_str}\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"Overall F1: {result['final_f1']:.4f}\n\n")

            subset = attacks_df[attacks_df['epsilon_str'] == eps_str]

            if len(subset) > 0:
                f.write("Detection Rates by Attack Type:\n")
                f.write("-" * 40 + "\n")

                for _, row in subset.iterrows():
                    f.write(f"  {row['attack_type']:15s}: {row['detection_rate']*100:5.1f}%\n")

                rates = subset['detection_rate'].values
                if len(rates) > 0 and max(rates) > 0:
                    f.write(f"\nFairness Metrics:\n")
                    f.write(f"  Min Detection: {min(rates)*100:.1f}%\n")
                    f.write(f"  Max Detection: {max(rates)*100:.1f}%\n")
                    f.write(f"  Gap: {(max(rates)-min(rates))*100:.1f}%\n")
                    f.write(f"  Disparate Impact: {min(rates)/max(rates):.3f}\n")

    print(f"✓ Saved: {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""

    print("\n" + "=" * 80)
    print("FAIRNESS VALIDATION FOR FL-NIDS (FIXED)")
    print("Direct final model evaluation")
    print("=" * 80 + "\n")

    # Setup
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y_binary, attack_categories = load_unsw_data()

    # Split
    X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
        X, y_binary, attack_categories,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y_binary
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\n✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")

    input_dim = X_train.shape[1]

    # Run experiments
    all_results = []

    for exp_config in config.EXPERIMENTS:
        result = run_experiment(X_train, y_train, X_test, y_test, cat_test, exp_config, input_dim)
        all_results.append(result)

        # Save
        np.savez(
            output_dir / f"predictions_{exp_config['name'].replace(' ', '_').replace('=', '')}.npz",
            y_pred=result['y_pred'],
            y_prob=result['y_prob'],
            y_true=result['y_true'],
            attack_categories=result['attack_categories']
        )

    # Analyze
    df = analyze_fairness(all_results, output_dir)

    if df is not None and len(df) > 0:
        create_visualizations(df, output_dir)
        generate_report(df, all_results, output_dir)

        print(f"\n{'='*80}")
        print("✓ FAIRNESS VALIDATION COMPLETE!")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print("⚠ FAIRNESS ANALYSIS INCOMPLETE")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
