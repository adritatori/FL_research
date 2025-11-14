"""
Q1-Quality NWDAF FL+DP Experiment with Pilot Study
===================================================
Federated Learning with Differential Privacy on UNSW-NB15 Dataset

MODES:
- PILOT: 16 experiments (~30-60 min) - Quick validation
- FOCUSED: 36 experiments (~2-3 hours) - Targeted analysis
- FULL: 108 experiments (~4-8 hours) - Complete grid

Author: Research Team
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)

import flwr as fl
from flwr.common import NDArrays, Scalar, Parameters, Metrics
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg

from opacus import PrivacyEngine

import psutil
from joblib import Parallel, delayed

# ============================================================================
# CONFIGURATION WITH PILOT STUDY SUPPORT
# ============================================================================

class ExperimentConfig:
    """Centralized configuration with pilot/focused/full modes"""

    # ========================================================================
    # EXPERIMENT MODE SELECTION
    # ========================================================================
    EXPERIMENT_MODE = 'pilot'  # Options: 'pilot', 'focused', 'full'

    # Paths
    DATA_PATH = '/content/drive/MyDrive/IDSDatasets/UNSW 15'
    RESULTS_DIR = './results'
    LOGS_DIR = './logs'

    # Random seeds
    RANDOM_SEED = 42

    # Dataset parameters
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1

    # Federated Learning parameters
    NUM_CLIENTS = 3
    CLIENT_FRACTION = 1.0
    MIN_FIT_CLIENTS = 2
    MIN_AVAILABLE_CLIENTS = 2

    # Local training
    LOCAL_EPOCHS = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001

    # Differential Privacy
    TARGET_DELTA = 1e-5
    MAX_GRAD_NORM = 1.0

    # Model parameters
    HIDDEN_DIMS = [128, 64, 32]
    DROPOUT_RATE = 0.2
    TRIM_RATIO = 0.1

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========================================================================
    # MODE-SPECIFIC CONFIGURATIONS
    # ========================================================================
    @classmethod
    def get_config(cls):
        """Return configuration based on EXPERIMENT_MODE"""

        if cls.EXPERIMENT_MODE == 'pilot':
            return {
                'name': 'PILOT STUDY',
                'description': 'Quick validation - identifies best configs',
                'num_rounds': 5,
                'epsilon_values': [float('inf')],  # DP vs no-DP
                'malicious_fractions': [0.0, 0.2],
                'attack_types': ['none', 'label_flip'],
                'aggregators': ['fedavg', 'trimmed_mean'],
                'num_runs': 1,
                'use_sample': True,
                'sample_fraction': 0.2,
                'total_experiments': 1,  # 2×2×2×2×2
                'estimated_time': '30-60 minutes'
            }

        elif cls.EXPERIMENT_MODE == 'focused':
            return {
                'name': 'FOCUSED STUDY',
                'description': 'Targeted analysis on promising configs',
                'num_rounds': 15,
                'epsilon_values': [float('inf')],
                'malicious_fractions': [0.0, 0.2],
                'attack_types': ['none', 'label_flip'],
                'aggregators': ['fedavg', 'trimmed_mean'],
                'num_runs': 3,
                'use_sample': True,
                'sample_fraction': 0.1,
                'total_experiments': 36,  # 3×2×2×2×3
                'estimated_time': '2-3 hours'
            }

        else:  # 'full'
            return {
                'name': 'FULL STUDY',
                'description': 'Complete experimental grid',
                'num_rounds': 20,
                'epsilon_values': [1.0, 5.0, float('inf')],
                'malicious_fractions': [0.0, 0.2],
                'attack_types': ['none', 'label_flip'],
                'aggregators': ['fedavg', 'trimmed_mean', 'median'],
                'num_runs': 3,
                'use_sample': False,
                'sample_fraction': 1.0,
                'total_experiments': 108,  # 3×2×2×3×3
                'estimated_time': '4-8 hours'
            }

    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        Path(cls.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.LOGS_DIR).mkdir(parents=True, exist_ok=True)

    @classmethod
    def set_seed(cls, seed: int = None):
        """Set random seeds for reproducibility"""
        seed = seed or cls.RANDOM_SEED
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @classmethod
    def print_mode_info(cls):
        """Print current mode configuration"""
        cfg = cls.get_config()
        print("\n" + "="*80)
        print(f"EXPERIMENT MODE: {cfg['name']}")
        print("="*80)
        print(f"Description: {cfg['description']}")
        print(f"Total experiments: {cfg['total_experiments']}")
        print(f"Estimated time: {cfg['estimated_time']}")
        print(f"\nParameters:")
        print(f"  Rounds: {cfg['num_rounds']}")
        print(f"  Epsilon values: {cfg['epsilon_values']}")
        print(f"  Malicious fractions: {cfg['malicious_fractions']}")
        print(f"  Attack types: {cfg['attack_types']}")
        print(f"  Aggregators: {cfg['aggregators']}")
        print(f"  Runs per config: {cfg['num_runs']}")
        if cfg['use_sample']:
            print(f"  Dataset sample: {cfg['sample_fraction']*100:.0f}%")
        print("="*80 + "\n")

config = ExperimentConfig()
config.setup_directories()
config.set_seed()

# ============================================================================
# DATA LOADING AND PREPROCESSING
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
# FEDERATED DATA PARTITIONING
# ============================================================================

class FederatedDataPartitioner:
    """Partition data across clients"""

    @staticmethod
    def iid_partition(X: np.ndarray, y: np.ndarray, num_clients: int) -> List[Tuple]:
        """IID partitioning - random split"""
        indices = np.random.permutation(len(X))
        splits = np.array_split(indices, num_clients)

        partitions = []
        for split in splits:
            partitions.append((X[split], y[split]))

        return partitions

# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class IntrusionDetectionMLP(nn.Module):
    """MLP for binary intrusion detection - FIXED"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.3):  # Increased dropout
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # ✅ Changed from GroupNorm
                nn.LeakyReLU(0.2),           # ✅ Changed from ReLU
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        # No sigmoid - BCEWithLogitsLoss handles it

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ============================================================================
# METRICS TRACKER
# ============================================================================

class MetricsTracker:
    """Track all experimental metrics"""

    def __init__(self):
        self.metrics = defaultdict(list)

    def log(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)

    def get_summary(self) -> Dict:
        summary = {}
        for key, values in self.metrics.items():
            if isinstance(values[0], (int, float)):
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        return summary

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2, default=str)

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
        
        # Calculate pos_weight correctly
        all_labels = []
        for _, labels in trainloader:
            all_labels.extend(labels.numpy().flatten())
        all_labels = np.array(all_labels)
        
        num_class_0 = np.sum(all_labels == 0)
        num_class_1 = np.sum(all_labels == 1)
        
        print(f"[Client {cid}] Training data: {num_class_0} class-0, {num_class_1} class-1")
        
        if num_class_1 > 0:
            pos_weight = torch.tensor([num_class_0 / num_class_1]).to(config.DEVICE)
        else:
            pos_weight = torch.tensor([1.0]).to(config.DEVICE)
        
        print(f"[Client {cid}] pos_weight = {pos_weight.item():.4f}")
            
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.privacy_engine = None

    def setup_dp(self, epsilon: float):
        """Setup differential privacy"""
        if epsilon == float('inf'):
            return

        self.privacy_engine = PrivacyEngine()

        cfg = self.config.get_config()
        self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.trainloader,
            target_epsilon=epsilon,
            target_delta=self.config.TARGET_DELTA,
            epochs=cfg['num_rounds'] * self.config.LOCAL_EPOCHS,
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
            for batch_idx, (data, target) in enumerate(self.trainloader):
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # DEBUG: First batch of first epoch
                if epoch == 0 and batch_idx == 0:
                    print(f"\n[Client {self.cid}] First training batch:")
                    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                    print(f"  Loss: {loss.item():.4f}")
                    print(f"  Target: {target.sum().item():.0f}/{len(target)} positives")
                
                loss.backward()
                
                # DEBUG: Check gradients
                if epoch == 0 and batch_idx == 0:
                    grad_norm = sum(p.grad.norm().item() for p in self.model.parameters() if p.grad is not None)
                    print(f"  Gradient norm: {grad_norm:.4f}")
                
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"[Client {self.cid}] Training done: avg_loss={avg_loss:.4f}")

        end_time = time.time()
        end_cpu = psutil.Process().cpu_times()

        updated_params = self.get_parameters(config={})
        param_size = sum(p.nbytes for p in updated_params)

        metrics = {
            'client_id': self.cid,
            'latency': end_time - start_time,
            'cpu_time': end_cpu.user - start_cpu.user,
            'bandwidth_bytes': param_size,
            'avg_loss': avg_loss
        }

        if self.privacy_engine:
            epsilon = self.privacy_engine.get_epsilon(delta=self.config.TARGET_DELTA)
            metrics['epsilon_spent'] = epsilon
        else:
            metrics['epsilon_spent'] = float('inf')

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

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5
            
        print(f"\n[DEBUG] Client {self.cid} Evaluation:")
        print(f"  Total samples: {len(all_labels)}")
        print(f"  Pred distribution: {np.bincount(np.array(all_preds).astype(int))}")
        print(f"  True distribution: {np.bincount(np.array(all_labels).astype(int))}")
        print(f"  Raw prob range: [{min(all_probs):.4f}, {max(all_probs):.4f}]")
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'loss': total_loss / len(self.testloader) if len(self.testloader) > 0 else 0
        }

        return float(total_loss / len(self.testloader)) if len(self.testloader) > 0 else 0.0, len(self.testloader.dataset), metrics

    def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0

        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
      
                logits = self.model(data)  # Raw logits
                output = torch.sigmoid(logits)  # Apply sigmoid for evaluation
                
                loss = self.criterion(logits, target)  # Loss uses raw logits
                total_loss += loss.item()

                probs = output.cpu().numpy()
                preds = (output > 0.5).float().cpu().numpy()

                all_probs.extend(probs.flatten())
                all_preds.extend(preds.flatten())
                all_labels.extend(target.cpu().numpy().flatten())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5
        print(f"\n[DEBUG] Client {self.cid} Evaluation:")
        print(f"  Total samples: {len(all_labels)}")
        print(f"  Pred distribution: {np.bincount(np.array(all_preds).astype(int))}")
        print(f"  True distribution: {np.bincount(np.array(all_labels).astype(int))}")
        print(f"  Raw prob range: [{min(all_probs):.4f}, {max(all_probs):.4f}]")
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'loss': total_loss / len(self.testloader) if len(self.testloader) > 0 else 0
        }

        return float(total_loss / len(self.testloader)) if len(self.testloader) > 0 else 0.0, len(self.testloader.dataset), metrics


class MaliciousClient(BaseClient):
    """Client that performs label flip attack"""

    def __init__(self, cid: int, model: nn.Module, trainloader: DataLoader,
                 testloader: DataLoader, config: ExperimentConfig, attack_type: str):
        super().__init__(cid, model, trainloader, testloader, config)
        self.attack_type = attack_type
        print(f"⚠️  Client {cid} is MALICIOUS (Attack: {attack_type})")

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
            'attack_type': 'label_flip'
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

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]]:

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

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]]:

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
# EXPERIMENT RUNNER (PARALLELIZED WITH JOBLIB)
# ============================================================================

class ExperimentRunner:
    """Run experiments in parallel with joblib"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = defaultdict(list)

    @staticmethod
    def _run_single_experiment(X_train, y_train, X_test, y_test, input_dim,
                               epsilon, alpha, attack_type, aggregator, run_id,
                               base_config):
        """Run a single experiment configuration"""

        cfg = base_config.get_config()
        ExperimentConfig.set_seed(base_config.RANDOM_SEED + run_id)

        partitions = FederatedDataPartitioner.iid_partition(
            X_train, y_train, base_config.NUM_CLIENTS
        )

        num_malicious = int(alpha * base_config.NUM_CLIENTS)
        malicious_ids = list(range(num_malicious))

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
            print(model)
            print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

            if client_id in malicious_ids and attack_type != 'none':
                return MaliciousClient(client_id, model, train_loader, test_loader,
                                       base_config, attack_type).to_client()
            else:
                client = BaseClient(client_id, model, train_loader, test_loader, base_config)
                if epsilon != float('inf'):
                    client.setup_dp(epsilon)
                return client.to_client()

        def weighted_avg(metrics: List[Tuple[int, Metrics]]) -> Metrics:
            # During fit, only aggregate available metrics
            if not metrics:
                return {}

            # Check what metrics are available
            sample_metrics = metrics[0][1]
            aggregated = {}

            for key in ['accuracy', 'f1', 'auc', 'latency', 'cpu_time']:
                if key in sample_metrics:
                    values = [m[key] * n for n, m in metrics]
                    total = sum(n for n, _ in metrics)
                    if total > 0:
                        aggregated[key] = sum(values) / total

            return aggregated

        strategy_params = {
            "fraction_fit": base_config.CLIENT_FRACTION,
            "fraction_evaluate": 1.0,
            "min_fit_clients": base_config.MIN_FIT_CLIENTS,
            "min_available_clients": base_config.MIN_AVAILABLE_CLIENTS,
            "evaluate_metrics_aggregation_fn": weighted_avg,
            "fit_metrics_aggregation_fn": weighted_avg,
        }

        if aggregator == 'trimmed_mean':
            strategy = TrimmedMeanStrategy(**strategy_params, trim_ratio=base_config.TRIM_RATIO)
        elif aggregator == 'median':
            strategy = MedianStrategy(**strategy_params)
        else:
            strategy = FedAvg(**strategy_params)

        start_exp_time = time.time()

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=base_config.NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=cfg['num_rounds']),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0}
        )

        total_exp_time = time.time() - start_exp_time

        final_metrics = {}

        if history.metrics_centralized and 'fit_metrics' in history.metrics_centralized:
            all_fit_metrics = [m for _, m_list in history.metrics_centralized['fit_metrics'] for _, m in m_list]
            all_latencies = [m['latency'] for m in all_fit_metrics if 'latency' in m]
            all_cpu_times = [m['cpu_time'] for m in all_fit_metrics if 'cpu_time' in m]
            final_metrics['avg_client_latency'] = np.mean(all_latencies) if all_latencies else 0.0
            final_metrics['avg_client_cpu'] = np.mean(all_cpu_times) if all_cpu_times else 0.0

        if history.metrics_distributed:
            for metric_name in ['accuracy', 'f1', 'auc']:
                if metric_name in history.metrics_distributed:
                    values = [v for _, v in history.metrics_distributed[metric_name]]
                    final_metrics[f'final_{metric_name}'] = values[-1] if values else 0.0
                    final_metrics[f'best_{metric_name}'] = max(values) if values else 0.0
                    final_metrics[f'avg_{metric_name}'] = np.mean(values) if values else 0.0

        result = {
            'epsilon': epsilon,
            'alpha': alpha,
            'attack_type': attack_type,
            'aggregator': aggregator,
            'num_rounds': cfg['num_rounds'],
            'total_time': total_exp_time,
            'avg_round_latency': total_exp_time / cfg['num_rounds'],
            **final_metrics,
            'run_id': run_id
        }

        print(f"  ✓ Finished: ε={epsilon}, α={alpha}, att={attack_type}, agg={aggregator} -> F1={result.get('final_f1', 0.0):.4f}")
        return result

    def run_full_experiment(self, X_train, y_train, X_test, y_test, input_dim: int):
        """Run experiments in parallel with joblib"""

        cfg = self.config.get_config()

        self.config.print_mode_info()

        start_time = time.time()

        print(f"Using joblib with {os.cpu_count()} CPUs 🚀\n")

        # Build task list
        tasks = []
        for run_id in range(cfg['num_runs']):
            for epsilon in cfg['epsilon_values']:
                for alpha in cfg['malicious_fractions']:
                    for attack_type in cfg['attack_types']:
                        if alpha == 0.0 and attack_type != 'none':
                            continue
                        if alpha > 0.0 and attack_type == 'none':
                            continue

                        for aggregator in cfg['aggregators']:
                            tasks.append((X_train, y_train, X_test, y_test, input_dim,
                                        epsilon, alpha, attack_type, aggregator, run_id, self.config))

        print(f"Total experiments to run: {len(tasks)}")
        print("Running experiments in parallel...\n")

        # Run in parallel with joblib
        # Replace Parallel section with:
        all_results = []
        for i, task in enumerate(tasks):
            print(f"Running experiment {i+1}/{len(tasks)}...")
            result = ExperimentRunner._run_single_experiment(*task)
            all_results.append(result)

        print(f"\n{'='*80}")
        print(f"COMPLETED {len(all_results)} EXPERIMENTS")
        print(f"{'='*80}\n")

        for result in all_results:
            if result:
                key = f"eps{result['epsilon']}_alpha{result['alpha']}_{result['attack_type']}_{result['aggregator']}"
                self.results[key].append(result)

        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"RUNTIME SUMMARY:")
        print(f"  Total runtime: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
        print(f"  Average time per experiment: {total_time/len(all_results):.1f} seconds")
        print(f"  Experiments completed: {len(all_results)}")
        print(f"{'='*80}\n")

        return self.results

    def save_results(self, filename: str = None):
        """Save results to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = self.config.EXPERIMENT_MODE
            filename = f"{self.config.RESULTS_DIR}/results_{mode}_{timestamp}.json"

        results_dict = {k: v for k, v in self.results.items()}

        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)

        print(f"✓ Results saved to: {filename}")
        return filename

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

class ResultsAnalyzer:
    """Analyze and visualize experimental results"""

    def __init__(self, results: Dict):
        self.results = results
        self.df = self._results_to_dataframe()

    def _results_to_dataframe(self) -> pd.DataFrame:
        rows = []
        for key, experiments in self.results.items():
            for exp in experiments:
                rows.append(exp)
        return pd.DataFrame(rows)

    def generate_summary_statistics(self):
        """Generate summary statistics"""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80 + "\n")

        if self.df.empty:
            print("No results to analyze.")
            return

        best_config = self.df.loc[self.df['final_f1'].idxmax()]
        print("🏆 BEST CONFIGURATION:")
        print(f"  ε = {best_config['epsilon']}")
        print(f"  α = {best_config['alpha']}")
        print(f"  Aggregator = {best_config['aggregator']}")
        print(f"  Attack = {best_config['attack_type']}")
        print(f"  F1 = {best_config['final_f1']:.4f}")
        print(f"  AUC = {best_config['final_auc']:.4f}")
        print(f"  Latency = {best_config['avg_round_latency']:.2f}s")

        print("\n📊 CONFIGURATION COMPARISON:")
        summary = self.df.groupby(['epsilon', 'alpha', 'aggregator', 'attack_type']).agg({
            'final_f1': ['mean', 'std'],
            'final_auc': ['mean', 'std'],
            'avg_round_latency': ['mean', 'std']
        }).round(4)
        print(summary.head(10))

    def plot_pilot_results(self, save_path: str = None):
        """Quick visualization for pilot study"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Pilot Study Results - {config.EXPERIMENT_MODE.upper()}',
                     fontsize=16, fontweight='bold')

        # F1 vs Epsilon
        ax = axes[0, 0]
        df_clean = self.df[(self.df['alpha'] == 0.0) & (self.df['attack_type'] == 'none')]
        if not df_clean.empty:
            for agg in df_clean['aggregator'].unique():
                df_agg = df_clean[df_clean['aggregator'] == agg]
                grouped = df_agg.groupby('epsilon')['final_f1'].agg(['mean', 'std']).reset_index()
                ax.errorbar(grouped['epsilon'], grouped['mean'], yerr=grouped['std'],
                           marker='o', label=agg, linewidth=2, markersize=8, capsize=5)
        ax.set_xlabel('Epsilon')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 vs Privacy Budget')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        # F1 vs Malicious Fraction
        ax = axes[0, 1]
        df_attack = self.df[self.df['attack_type'] == 'label_flip']
        if not df_attack.empty:
            for agg in df_attack['aggregator'].unique():
                df_agg = df_attack[df_attack['aggregator'] == agg]
                grouped = df_agg.groupby('alpha')['final_f1'].agg(['mean', 'std']).reset_index()
                ax.errorbar(grouped['alpha'], grouped['mean'], yerr=grouped['std'],
                           marker='s', label=agg, linewidth=2, markersize=8, capsize=5)
        ax.set_xlabel('Malicious Fraction')
        ax.set_ylabel('F1 Score')
        ax.set_title('Robustness Under Attack')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Latency Comparison
        ax = axes[1, 0]
        if not self.df.empty:
            grouped = self.df.groupby('aggregator')['avg_round_latency'].agg(['mean', 'std']).reset_index()
            ax.bar(grouped['aggregator'], grouped['mean'], yerr=grouped['std'],
                   capsize=5, alpha=0.7)
        ax.set_xlabel('Aggregator')
        ax.set_ylabel('Avg Round Latency (s)')
        ax.set_title('Operational Costs')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Heatmap
        ax = axes[1, 1]
        pivot = self.df.pivot_table(values='final_f1',
                                     index='aggregator',
                                     columns='epsilon',
                                     aggfunc='mean')
        if not pivot.empty:
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                        vmin=0, vmax=1, ax=ax)
        ax.set_title('F1 Score Heatmap')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    print("\n" + "="*80)
    print("Q1-QUALITY NWDAF FL+DP EXPERIMENT WITH PILOT STUDY")
    print("Dataset: UNSW-NB15")
    print("="*80 + "\n")

    try:
        cfg = config.get_config()

        print("STEP 1: Loading data...")
        data_loader = UNSWDataLoader(config.DATA_PATH)
        X, y, feature_names = data_loader.load_and_preprocess(
            use_sample=cfg['use_sample'],
            sample_fraction=cfg['sample_fraction']
        )

        print("\nSTEP 2: Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        print(f"✓ Training set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")

        input_dim = X_train.shape[1]

        print("\nSTEP 3: Running experiments...")
        runner = ExperimentRunner(config)
        results = runner.run_full_experiment(X_train, y_train, X_test, y_test, input_dim)

        print("\nSTEP 4: Saving results...")
        results_file = runner.save_results()

        print("\nSTEP 5: Analyzing results...")
        analyzer = ResultsAnalyzer(results)

        analyzer.plot_pilot_results(
            save_path=f"{config.RESULTS_DIR}/{config.EXPERIMENT_MODE}_results.png"
        )

        analyzer.generate_summary_statistics()

        print("\n" + "="*80)
        print("✅ EXPERIMENT COMPLETE!")
        print("="*80)
        print(f"\nResults saved in: {config.RESULTS_DIR}/")
        print(f"Mode: {cfg['name']}")
        print(f"\nNext steps:")
        if config.EXPERIMENT_MODE == 'pilot':
            print("  1. Review results to identify best configs")
            print("  2. Change EXPERIMENT_MODE to 'focused' for deeper analysis")
            print("  3. Run 'focused' experiments on promising configurations")
        elif config.EXPERIMENT_MODE == 'focused':
            print("  1. Validate findings from focused study")
            print("  2. Change EXPERIMENT_MODE to 'full' for final paper")
            print("  3. Run 'full' experiments for comprehensive results")
        else:
            print("  📊 Use these results for your Q1 paper!")

    except FileNotFoundError as e:
        print(f"\n❌ CRITICAL ERROR: Could not find data file.")
        print(f"  Details: {e}")
        print(f"  Please update DATA_PATH in ExperimentConfig class.")
    except Exception as e:
        print(f"\n❌ AN UNEXPECTED ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
