"""
Single Experiment Runner
========================
Executes a single FL-NIDS experiment configuration.

Keeps existing Flower + Opacus integration intact.
Adds comprehensive metrics tracking and progress reporting.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import psutil
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

import flwr as fl
from flwr.common import NDArrays, Scalar, Parameters, Metrics
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from opacus import PrivacyEngine

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from config import ExperimentConfig
from metrics import MetricsTracker, compute_comprehensive_metrics, aggregate_client_metrics


# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class IntrusionDetectionMLP(nn.Module):
    """MLP for binary intrusion detection"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
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
    """Base client with DP and metrics tracking"""

    def __init__(self, cid: int, model: nn.Module, trainloader: DataLoader,
                 testloader: DataLoader, config: ExperimentConfig):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.config = config

        # Calculate pos_weight for imbalanced classes
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
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.privacy_engine = None

    def setup_dp(self, epsilon: float, num_rounds: int):
        """Setup differential privacy"""
        if epsilon == float('inf'):
            return

        self.privacy_engine = PrivacyEngine()

        self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.trainloader,
            target_epsilon=epsilon,
            target_delta=self.config.TARGET_DELTA,
            epochs=num_rounds * self.config.LOCAL_EPOCHS,
            max_grad_norm=self.config.MAX_GRAD_NORM,
        )

    def get_parameters(self, config: Dict) -> NDArrays:
        """Return model parameters as list of NumPy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays):
        """Set model parameters from list of NumPy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
        """Train the model on local data"""
        self.set_parameters(parameters)
        self.model.train()

        start_time = time.time()
        start_cpu = psutil.Process().cpu_times()

        total_loss = 0
        num_batches = 0

        for epoch in range(self.config.LOCAL_EPOCHS):
            for data, target in self.trainloader:
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        end_time = time.time()
        end_cpu = psutil.Process().cpu_times()

        updated_params = self.get_parameters(config={})
        param_size = sum(p.nbytes for p in updated_params)

        metrics = {
            'client_id': self.cid,
            'latency': end_time - start_time,
            'cpu_time': end_cpu.user - start_cpu.user,
            'bandwidth_bytes': param_size,
            'avg_loss': avg_loss,
            'num_examples': len(self.trainloader.dataset),
        }

        if self.privacy_engine:
            epsilon = self.privacy_engine.get_epsilon(delta=self.config.TARGET_DELTA)
            metrics['epsilon_consumed'] = epsilon

        return updated_params, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the model on test data"""
        self.set_parameters(parameters)
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0

        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)

                logits = self.model(data)
                output = torch.sigmoid(logits)

                loss = self.criterion(logits, target)
                total_loss += loss.item()

                probs = output.cpu().numpy()
                preds = (output > 0.5).float().cpu().numpy()

                all_probs.extend(probs.flatten())
                all_preds.extend(preds.flatten())
                all_labels.extend(target.cpu().numpy().flatten())

        # Compute comprehensive metrics
        metrics = compute_comprehensive_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )

        metrics['loss'] = total_loss / len(self.testloader) if len(self.testloader) > 0 else 0
        metrics['num_examples'] = len(all_labels)

        return float(metrics['loss']), len(self.testloader.dataset), metrics


class MaliciousClient(BaseClient):
    """Client that performs label flip attack"""

    def __init__(self, cid: int, model: nn.Module, trainloader: DataLoader,
                 testloader: DataLoader, config: ExperimentConfig, attack_type: str):
        super().__init__(cid, model, trainloader, testloader, config)
        self.attack_type = attack_type

    def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
        if self.attack_type == 'label_flip':
            return self._label_flip_attack(parameters, config)
        else:
            return super().fit(parameters, config)

    def _label_flip_attack(self, parameters: NDArrays, config: Dict):
        self.set_parameters(parameters)
        self.model.train()

        start_time = time.time()

        for epoch in range(self.config.LOCAL_EPOCHS):
            for data, target in self.trainloader:
                flipped_target = 1 - target
                data = data.to(self.config.DEVICE)
                flipped_target = flipped_target.to(self.config.DEVICE)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, flipped_target)
                loss.backward()
                self.optimizer.step()

        latency = time.time() - start_time
        updated_params = self.get_parameters(config={})

        metrics = {
            'client_id': self.cid,
            'latency': latency,
            'attack_type': 'label_flip',
            'num_examples': len(self.trainloader.dataset),
        }

        return updated_params, len(self.trainloader.dataset), metrics


# ============================================================================
# ROBUST AGGREGATION STRATEGIES
# ============================================================================

class TrimmedMeanStrategy(FedAvg):
    """Trimmed mean aggregation"""

    def __init__(self, trim_ratio: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.trim_ratio = trim_ratio

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        weights = [w for w, _ in weights_results]
        aggregated = []
        trim_count = int(self.trim_ratio * len(weights))

        for layer_idx in range(len(weights[0])):
            layer_weights = np.stack([w[layer_idx] for w in weights])

            if trim_count > 0 and trim_count < len(weights) // 2:
                sorted_weights = np.sort(layer_weights, axis=0)
                trimmed = sorted_weights[trim_count:-trim_count]
                aggregated_layer = np.mean(trimmed, axis=0)
            else:
                aggregated_layer = np.mean(layer_weights, axis=0)

            aggregated.append(aggregated_layer)

        parameters_aggregated = ndarrays_to_parameters(aggregated)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated


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
# DATA PARTITIONING
# ============================================================================

def iid_partition(X: np.ndarray, y: np.ndarray, num_clients: int) -> List[Tuple]:
    """IID partitioning - random split"""
    indices = np.random.permutation(len(X))
    splits = np.array_split(indices, num_clients)

    partitions = []
    for split in splits:
        partitions.append((X[split], y[split]))

    return partitions


# ============================================================================
# SINGLE EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    experiment_config: Dict,
    run_id: int,
    base_config: ExperimentConfig
) -> MetricsTracker:
    """
    Run a single FL-NIDS experiment

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        input_dim: Number of input features
        experiment_config: Experiment configuration
        run_id: Run ID for seed
        base_config: Base configuration object

    Returns:
        MetricsTracker with results
    """

    # Set seed for reproducibility
    seed = base_config.RANDOM_SEED + run_id * 1000
    base_config.set_seed(seed)

    # Initialize metrics tracker
    tracker = MetricsTracker(experiment_config)

    # Print experiment info
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting Experiment:")
    print(f"  Config: ε={experiment_config['epsilon']}, "
          f"Agg={experiment_config['aggregator']}, "
          f"Attack={experiment_config['attack_type']} ({experiment_config['attack_ratio']*100:.0f}%)")
    print(f"  Run: {run_id + 1}/{base_config.NUM_RUNS}")
    print(f"  Seed: {seed}")

    # Partition data
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Partitioning data across {base_config.NUM_CLIENTS} clients...")
    partitions = iid_partition(X_train, y_train, base_config.NUM_CLIENTS)

    # Determine malicious clients
    num_malicious = int(experiment_config['attack_ratio'] * base_config.NUM_CLIENTS)
    malicious_ids = list(range(num_malicious))

    if num_malicious > 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  {num_malicious} clients are malicious")

    # Client function
    def client_fn(cid: str):
        client_id = int(cid)
        X_client, y_client = partitions[client_id]

        train_dataset = TensorDataset(
            torch.FloatTensor(X_client),
            torch.FloatTensor(y_client).view(-1, 1)
        )
        train_loader = DataLoader(train_dataset, batch_size=base_config.BATCH_SIZE, shuffle=True)

        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test).view(-1, 1)
        )
        test_loader = DataLoader(test_dataset, batch_size=base_config.BATCH_SIZE)

        model = IntrusionDetectionMLP(
            input_dim=input_dim,
            hidden_dims=base_config.HIDDEN_DIMS,
            dropout=base_config.DROPOUT_RATE
        ).to(base_config.DEVICE)

        if client_id in malicious_ids and experiment_config['attack_type'] != 'none':
            return MaliciousClient(
                client_id, model, train_loader, test_loader,
                base_config, experiment_config['attack_type']
            ).to_client()
        else:
            client = BaseClient(client_id, model, train_loader, test_loader, base_config)
            if experiment_config['epsilon'] != float('inf'):
                num_rounds = base_config.get_phase_config()['num_rounds']
                client.setup_dp(experiment_config['epsilon'], num_rounds)
            return client.to_client()

    # Metrics aggregation function
    def weighted_avg(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        if not metrics:
            return {}

        sample_metrics = metrics[0][1]
        aggregated = {}

        for key in ['accuracy', 'f1', 'auc_roc', 'auc_pr', 'loss', 'latency', 'cpu_time']:
            if key in sample_metrics:
                values = [m[key] * n for n, m in metrics]
                total = sum(n for n, _ in metrics)
                if total > 0:
                    aggregated[key] = sum(values) / total

        # Max epsilon
        epsilon_values = [m.get('epsilon_consumed', 0.0) for _, m in metrics if 'epsilon_consumed' in m]
        if epsilon_values:
            aggregated['epsilon_consumed'] = max(epsilon_values)

        return aggregated

    # Select strategy
    strategy_params = {
        "fraction_fit": base_config.CLIENT_FRACTION,
        "fraction_evaluate": 1.0,
        "min_fit_clients": base_config.MIN_FIT_CLIENTS,
        "min_available_clients": base_config.MIN_AVAILABLE_CLIENTS,
        "evaluate_metrics_aggregation_fn": weighted_avg,
        "fit_metrics_aggregation_fn": weighted_avg,
    }

    if experiment_config['aggregator'] == 'trimmed_mean':
        strategy = TrimmedMeanStrategy(**strategy_params, trim_ratio=base_config.TRIM_RATIO)
    elif experiment_config['aggregator'] == 'median':
        strategy = MedianStrategy(**strategy_params)
    else:
        strategy = FedAvg(**strategy_params)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Strategy: {experiment_config['aggregator']}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting FL training...")

    # Run FL simulation
    start_time = time.time()
    num_rounds = base_config.get_phase_config()['num_rounds']

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=base_config.NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
    )

    tracker.total_time = time.time() - start_time

    # Extract metrics from history
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing results...")

    if history.metrics_distributed:
        for round_num in range(1, num_rounds + 1):
            round_metrics = {
                'round': round_num,
                'timestamp': time.time() - start_time,
            }

            # Extract metrics for this round
            for metric_name in ['accuracy', 'f1', 'auc_roc', 'loss']:
                if metric_name in history.metrics_distributed:
                    values = history.metrics_distributed[metric_name]
                    if len(values) >= round_num:
                        round_metrics[metric_name] = values[round_num - 1][1]

            tracker.log_round(round_num, round_metrics)

            # Print progress every 10 rounds or last round
            if round_num % 10 == 0 or round_num == num_rounds:
                f1 = round_metrics.get('f1', 0.0)
                loss = round_metrics.get('loss', 0.0)
                acc = round_metrics.get('accuracy', 0.0)
                time_elapsed = round_metrics['timestamp']
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Round {round_num}/{num_rounds}: "
                      f"F1={f1:.4f}, Loss={loss:.4f}, Acc={acc:.4f}, Time={time_elapsed:.1f}s")

            # Check for convergence
            if tracker.convergence_round == round_num:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Converged at round {round_num} (F1={f1:.4f})")

    summary = tracker.get_summary()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Experiment complete")
    print(f"  Final F1: {summary['final_f1']:.4f}")
    print(f"  Best F1: {summary['best_f1']:.4f} (round {summary['best_f1_round']})")
    print(f"  Total time: {summary['total_time']:.1f}s")
    if summary.get('converged'):
        print(f"  Converged: Round {summary['convergence_round']}")

    return tracker
