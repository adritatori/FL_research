"""
Metrics Tracking System
=======================
Comprehensive metrics collection for FL-NIDS experiments.

Tracks:
- Per-round metrics (loss, accuracy, F1, AUC, etc.)
- Per-class metrics (for 10 UNSW-NB15 categories)
- Privacy metrics (epsilon consumed, noise)
- Performance metrics (latency, convergence)
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)


class MetricsTracker:
    """Comprehensive metrics tracking for a single experiment"""

    def __init__(self, config: Dict):
        """Initialize metrics tracker

        Args:
            config: Experiment configuration dict
        """
        self.config = config
        self.round_metrics = []
        self.convergence_round = None
        self.early_stop_round = None
        self.total_time = 0.0
        self.start_time = None

    def log_round(self, round_num: int, metrics: Dict):
        """Log metrics for a single round

        Args:
            round_num: Current round number
            metrics: Dictionary of metrics for this round
        """
        round_data = {
            'round': round_num,
            'timestamp': metrics.get('timestamp', 0.0),
            **metrics
        }
        self.round_metrics.append(round_data)

        # Check for convergence
        if self.convergence_round is None:
            f1 = metrics.get('f1', 0.0)
            from config import config as cfg
            if f1 >= cfg.CONVERGENCE_THRESHOLD:
                self.convergence_round = round_num

    def check_early_stopping(self, patience: int = 10) -> bool:
        """Check if training should stop early

        Args:
            patience: Number of rounds without improvement

        Returns:
            True if should stop
        """
        if len(self.round_metrics) < patience + 1:
            return False

        # Get last `patience` F1 scores
        recent_f1 = [m['f1'] for m in self.round_metrics[-patience:]]
        best_f1 = max(recent_f1)

        # Check if no improvement in last `patience` rounds
        earlier_f1 = [m['f1'] for m in self.round_metrics[:-patience]]
        if earlier_f1:
            prev_best = max(earlier_f1)
            if best_f1 <= prev_best:
                self.early_stop_round = len(self.round_metrics)
                return True

        return False

    def get_summary(self) -> Dict:
        """Get summary statistics for the experiment

        Returns:
            Dictionary with summary metrics
        """
        if not self.round_metrics:
            return {}

        # Extract metric series
        f1_series = [m['f1'] for m in self.round_metrics]
        loss_series = [m.get('loss', 0.0) for m in self.round_metrics]
        acc_series = [m.get('accuracy', 0.0) for m in self.round_metrics]
        auc_series = [m.get('auc_roc', 0.0) for m in self.round_metrics]

        summary = {
            # Config
            'config': self.config,

            # Final metrics (last round)
            'final_round': len(self.round_metrics),
            'final_f1': f1_series[-1] if f1_series else 0.0,
            'final_accuracy': acc_series[-1] if acc_series else 0.0,
            'final_loss': loss_series[-1] if loss_series else 0.0,
            'final_auc_roc': auc_series[-1] if auc_series else 0.0,

            # Best metrics (across all rounds)
            'best_f1': max(f1_series) if f1_series else 0.0,
            'best_f1_round': int(np.argmax(f1_series)) + 1 if f1_series else 0,
            'best_accuracy': max(acc_series) if acc_series else 0.0,
            'best_auc_roc': max(auc_series) if auc_series else 0.0,

            # Average metrics
            'avg_f1': float(np.mean(f1_series)) if f1_series else 0.0,
            'avg_accuracy': float(np.mean(acc_series)) if acc_series else 0.0,
            'avg_loss': float(np.mean(loss_series)) if loss_series else 0.0,

            # Convergence
            'convergence_round': self.convergence_round,
            'converged': self.convergence_round is not None,
            'early_stop_round': self.early_stop_round,

            # Performance
            'total_time': self.total_time,
            'avg_round_time': self.total_time / len(self.round_metrics) if self.round_metrics else 0.0,
        }

        # Add privacy metrics if available
        if self.round_metrics and 'epsilon_consumed' in self.round_metrics[-1]:
            summary['final_epsilon'] = self.round_metrics[-1]['epsilon_consumed']

        return summary

    def save(self, filepath: Path):
        """Save metrics to JSON file

        Args:
            filepath: Path to save file
        """
        data = {
            'config': self.config,
            'summary': self.get_summary(),
            'round_metrics': self.round_metrics,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def load(filepath: Path) -> 'MetricsTracker':
        """Load metrics from JSON file

        Args:
            filepath: Path to load file

        Returns:
            MetricsTracker instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        tracker = MetricsTracker(data['config'])
        tracker.round_metrics = data['round_metrics']
        tracker.convergence_round = data['summary'].get('convergence_round')
        tracker.early_stop_round = data['summary'].get('early_stop_round')
        tracker.total_time = data['summary'].get('total_time', 0.0)

        return tracker


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                   y_prob: Optional[np.ndarray] = None) -> Dict:
    """Compute comprehensive evaluation metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # ROC AUC (requires probabilities)
    if y_prob is not None:
        try:
            metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics['auc_roc'] = 0.5

        # Precision-Recall AUC
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            metrics['auc_pr'] = float(auc(recall, precision))
        except ValueError:
            metrics['auc_pr'] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)

        # Additional metrics
        if tp + fp > 0:
            metrics['precision_manual'] = float(tp / (tp + fp))
        if tp + fn > 0:
            metrics['recall_manual'] = float(tp / (tp + fn))
        if tn + fp > 0:
            metrics['specificity'] = float(tn / (tn + fp))

    return metrics


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                               num_classes: int = 10) -> Dict:
    """Compute per-class metrics for multi-class classification

    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes

    Returns:
        Dictionary of per-class metrics
    """
    per_class = {}

    for class_id in range(num_classes):
        # Binary mask for this class
        mask_true = (y_true == class_id)
        mask_pred = (y_pred == class_id)

        if mask_true.sum() == 0:
            continue

        per_class[f'class_{class_id}'] = {
            'precision': float(precision_score(mask_true, mask_pred, zero_division=0)),
            'recall': float(recall_score(mask_true, mask_pred, zero_division=0)),
            'f1': float(f1_score(mask_true, mask_pred, zero_division=0)),
            'support': int(mask_true.sum()),
        }

    return per_class


def aggregate_client_metrics(client_metrics: List[Dict]) -> Dict:
    """Aggregate metrics from multiple clients

    Args:
        client_metrics: List of metric dictionaries from clients

    Returns:
        Aggregated metrics
    """
    if not client_metrics:
        return {}

    # Weighted average by number of examples
    total_examples = sum(m.get('num_examples', 1) for m in client_metrics)

    aggregated = {}

    # Metrics to aggregate
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr', 'loss']

    for key in metric_keys:
        values = [m.get(key, 0.0) for m in client_metrics if key in m]
        if values:
            weights = [m.get('num_examples', 1) for m in client_metrics if key in m]
            aggregated[key] = float(np.average(values, weights=weights))

    # Performance metrics (simple average)
    for key in ['latency', 'cpu_time']:
        values = [m.get(key, 0.0) for m in client_metrics if key in m]
        if values:
            aggregated[key] = float(np.mean(values))

    # Privacy metrics (max epsilon consumed)
    epsilon_values = [m.get('epsilon_consumed', 0.0) for m in client_metrics if 'epsilon_consumed' in m]
    if epsilon_values:
        aggregated['epsilon_consumed'] = float(max(epsilon_values))

    return aggregated


def format_metrics_for_display(metrics: Dict) -> str:
    """Format metrics for console display

    Args:
        metrics: Dictionary of metrics

    Returns:
        Formatted string
    """
    parts = []

    if 'loss' in metrics:
        parts.append(f"Loss={metrics['loss']:.4f}")
    if 'f1' in metrics:
        parts.append(f"F1={metrics['f1']:.4f}")
    if 'accuracy' in metrics:
        parts.append(f"Acc={metrics['accuracy']:.4f}")
    if 'auc_roc' in metrics:
        parts.append(f"AUC={metrics['auc_roc']:.4f}")
    if 'latency' in metrics:
        parts.append(f"Time={metrics['latency']:.1f}s")

    return ", ".join(parts)
