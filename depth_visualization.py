#!/usr/bin/env python3
"""
Depth Visualization Module for Federated Learning Phases

Provides comprehensive visualizations for analyzing model depth characteristics
across different FL training phases (baseline, privacy, aggregators, attacks, full).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class DepthVisualizer:
    """Visualizes model depth characteristics across FL phases."""

    def __init__(self, results_dir: str = "results", output_dir: str = "depth_visualizations"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Phase configurations
        self.phases = {
            'baseline': {'name': 'Baseline Study', 'color': '#2ecc71'},
            'privacy': {'name': 'Privacy Analysis', 'color': '#3498db'},
            'aggregators': {'name': 'Aggregator Robustness', 'color': '#e74c3c'},
            'attacks': {'name': 'Attack Scenarios', 'color': '#f39c12'},
            'full': {'name': 'Full Experimental Grid', 'color': '#9b59b6'}
        }

        # Model depth configurations to test
        self.depth_configs = {
            'shallow': [32],
            'medium': [64, 32],
            'deep': [128, 64, 32],
            'very_deep': [256, 128, 64, 32]
        }

    def create_model_with_depth(self, input_dim: int, hidden_dims: List[int],
                                  dropout: float = 0.1) -> nn.Module:
        """Create IntrusionDetectionMLP with specified depth."""
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

        class Model(nn.Module):
            def __init__(self, network):
                super().__init__()
                self.network = nn.Sequential(*network)

            def forward(self, x):
                return self.network(x)

        return Model(layers)

    def visualize_layer_weights_distribution(self,
                                             models: Dict[str, nn.Module],
                                             phase: str,
                                             save: bool = True) -> plt.Figure:
        """
        Visualize weight distributions across layers for different depth configurations.

        Args:
            models: Dict mapping depth name to model
            phase: Current phase name
            save: Whether to save the figure
        """
        num_depths = len(models)
        fig, axes = plt.subplots(num_depths, 1, figsize=(14, 4 * num_depths))
        if num_depths == 1:
            axes = [axes]

        fig.suptitle(f'Layer Weight Distributions - {self.phases[phase]["name"]}',
                    fontsize=16, fontweight='bold')

        for idx, (depth_name, model) in enumerate(models.items()):
            ax = axes[idx]
            weight_data = []
            layer_names = []

            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    weights = param.data.cpu().numpy().flatten()
                    weight_data.append(weights)
                    layer_names.append(name.replace('.weight', '').replace('network.', 'L'))

            if weight_data:
                positions = list(range(len(weight_data)))
                parts = ax.violinplot(weight_data, positions=positions,
                                     showmeans=True, showmedians=True)

                for pc in parts['bodies']:
                    pc.set_facecolor(self.phases[phase]['color'])
                    pc.set_alpha(0.7)

                ax.set_xticks(positions)
                ax.set_xticklabels(layer_names, rotation=45, ha='right')
                ax.set_ylabel('Weight Value')
                ax.set_title(f'{depth_name.upper()} - Hidden Dims: {self.depth_configs[depth_name]}')
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'{phase}_weight_distributions.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

        return fig

    def visualize_gradient_flow(self,
                                models: Dict[str, nn.Module],
                                X_sample: torch.Tensor,
                                y_sample: torch.Tensor,
                                phase: str,
                                save: bool = True) -> plt.Figure:
        """
        Visualize gradient flow through network layers for different depths.

        Args:
            models: Dict mapping depth name to model
            X_sample: Sample input data
            y_sample: Sample labels
            phase: Current phase name
            save: Whether to save the figure
        """
        num_depths = len(models)
        fig, axes = plt.subplots(1, num_depths, figsize=(5 * num_depths, 6))
        if num_depths == 1:
            axes = [axes]

        fig.suptitle(f'Gradient Flow Analysis - {self.phases[phase]["name"]}',
                    fontsize=16, fontweight='bold')

        criterion = nn.BCEWithLogitsLoss()

        for idx, (depth_name, model) in enumerate(models.items()):
            ax = axes[idx]
            model.train()
            model.zero_grad()

            # Forward pass
            output = model(X_sample)
            loss = criterion(output.squeeze(), y_sample.float())
            loss.backward()

            # Collect gradients
            grad_norms = []
            layer_names = []

            for name, param in model.named_parameters():
                if param.grad is not None and 'weight' in name:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    layer_names.append(name.replace('.weight', '').replace('network.', 'L'))

            if grad_norms:
                bars = ax.bar(range(len(grad_norms)), grad_norms,
                            color=self.phases[phase]['color'], alpha=0.7)

                # Add gradient values on bars
                for bar, val in zip(bars, grad_norms):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.4f}', ha='center', va='bottom', fontsize=8)

                ax.set_xticks(range(len(grad_norms)))
                ax.set_xticklabels(layer_names, rotation=45, ha='right')
                ax.set_ylabel('Gradient Norm')
                ax.set_title(f'{depth_name.upper()}\nHidden: {self.depth_configs[depth_name]}')
                ax.set_yscale('log')

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'{phase}_gradient_flow.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

        return fig

    def visualize_activation_statistics(self,
                                        models: Dict[str, nn.Module],
                                        X_sample: torch.Tensor,
                                        phase: str,
                                        save: bool = True) -> plt.Figure:
        """
        Visualize activation statistics at each layer for different depths.

        Args:
            models: Dict mapping depth name to model
            X_sample: Sample input data
            phase: Current phase name
            save: Whether to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Activation Statistics - {self.phases[phase]["name"]}',
                    fontsize=16, fontweight='bold')

        # Metrics to track
        metrics = {
            'mean': {},
            'std': {},
            'sparsity': {},  # % of zero/near-zero activations
            'max': {}
        }

        for depth_name, model in models.items():
            model.eval()
            activations = []

            # Register hooks to capture activations
            def get_activation(name):
                def hook(model, input, output):
                    activations.append(output.detach().cpu().numpy())
                return hook

            hooks = []
            for name, layer in model.network.named_modules():
                if isinstance(layer, nn.ReLU):
                    hooks.append(layer.register_forward_hook(get_activation(name)))

            # Forward pass
            with torch.no_grad():
                _ = model(X_sample)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            # Calculate statistics
            means = []
            stds = []
            sparsities = []
            maxes = []

            for act in activations:
                means.append(np.mean(act))
                stds.append(np.std(act))
                sparsities.append(np.mean(act < 0.01) * 100)
                maxes.append(np.max(act))

            metrics['mean'][depth_name] = means
            metrics['std'][depth_name] = stds
            metrics['sparsity'][depth_name] = sparsities
            metrics['max'][depth_name] = maxes

        # Plot each metric
        metric_titles = ['Mean Activation', 'Std Deviation', 'Sparsity (%)', 'Max Activation']
        metric_keys = ['mean', 'std', 'sparsity', 'max']

        for ax, title, key in zip(axes.flat, metric_titles, metric_keys):
            for depth_name, values in metrics[key].items():
                if values:
                    ax.plot(range(1, len(values) + 1), values,
                           marker='o', label=depth_name, linewidth=2, markersize=8)

            ax.set_xlabel('Layer Index')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'{phase}_activation_statistics.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

        return fig

    def visualize_depth_convergence_comparison(self,
                                               results_by_depth: Dict[str, List[Dict]],
                                               phase: str,
                                               save: bool = True) -> plt.Figure:
        """
        Compare convergence patterns across different model depths.

        Args:
            results_by_depth: Dict mapping depth name to list of round results
            phase: Current phase name
            save: Whether to save the figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Convergence Analysis by Depth - {self.phases[phase]["name"]}',
                    fontsize=16, fontweight='bold')

        metrics = ['f1', 'loss', 'accuracy', 'precision', 'recall', 'auc_roc']
        metric_titles = ['F1 Score', 'Training Loss', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC']

        for ax, metric, title in zip(axes.flat, metrics, metric_titles):
            for depth_name, rounds in results_by_depth.items():
                if rounds:
                    rounds_sorted = sorted(rounds, key=lambda x: x.get('round', 0))
                    x_vals = [r.get('round', i) for i, r in enumerate(rounds_sorted)]
                    y_vals = [r.get(metric, 0) for r in rounds_sorted]

                    ax.plot(x_vals, y_vals, marker='o', label=depth_name,
                           linewidth=2, markersize=6, alpha=0.8)

            ax.set_xlabel('Round')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            if metric == 'f1':
                ax.axhline(y=0.90, color='red', linestyle='--',
                          alpha=0.5, label='Convergence Threshold')

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'{phase}_depth_convergence.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

        return fig

    def visualize_privacy_depth_tradeoff(self,
                                         privacy_results: Dict[str, Dict[str, float]],
                                         phase: str = 'privacy',
                                         save: bool = True) -> plt.Figure:
        """
        Visualize privacy-utility tradeoff across different depths.

        Args:
            privacy_results: Dict mapping depth name to dict of epsilon -> f1_score
            phase: Current phase name
            save: Whether to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Privacy-Depth Tradeoff - {self.phases[phase]["name"]}',
                    fontsize=16, fontweight='bold')

        # Line plot
        ax = axes[0]
        for depth_name, eps_results in privacy_results.items():
            epsilons = sorted([e for e in eps_results.keys() if e != float('inf')])
            f1_scores = [eps_results[eps] for eps in epsilons]

            ax.plot(epsilons, f1_scores, marker='s', label=depth_name,
                   linewidth=2, markersize=10)

        ax.set_xlabel('Privacy Budget (ε)')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score vs Privacy Budget')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Heatmap
        ax = axes[1]
        depth_names = list(privacy_results.keys())
        epsilons = sorted(list(set().union(*[set(d.keys()) for d in privacy_results.values()])))
        epsilons = [e for e in epsilons if e != float('inf')]

        heatmap_data = np.zeros((len(depth_names), len(epsilons)))
        for i, depth in enumerate(depth_names):
            for j, eps in enumerate(epsilons):
                heatmap_data[i, j] = privacy_results[depth].get(eps, 0)

        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=[f'ε={e}' for e in epsilons],
                   yticklabels=depth_names, ax=ax)
        ax.set_title('F1 Score Heatmap (Depth × Privacy)')
        ax.set_xlabel('Privacy Budget')
        ax.set_ylabel('Model Depth')

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'{phase}_privacy_depth_tradeoff.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

        return fig

    def visualize_attack_robustness_by_depth(self,
                                             attack_results: Dict[str, Dict[float, float]],
                                             phase: str = 'attacks',
                                             save: bool = True) -> plt.Figure:
        """
        Visualize attack robustness across different model depths.

        Args:
            attack_results: Dict mapping depth name to dict of attack_ratio -> f1_score
            phase: Current phase name
            save: Whether to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Attack Robustness by Depth - {self.phases[phase]["name"]}',
                    fontsize=16, fontweight='bold')

        # Line plot
        ax = axes[0]
        for depth_name, ratio_results in attack_results.items():
            ratios = sorted(ratio_results.keys())
            f1_scores = [ratio_results[r] for r in ratios]

            ax.plot(ratios, f1_scores, marker='o', label=depth_name,
                   linewidth=2, markersize=10)

        ax.set_xlabel('Malicious Client Ratio')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score vs Attack Intensity')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Degradation bar chart
        ax = axes[1]
        degradations = {}
        for depth_name, ratio_results in attack_results.items():
            ratios = sorted(ratio_results.keys())
            if len(ratios) >= 2:
                baseline = ratio_results[ratios[0]]
                worst = ratio_results[ratios[-1]]
                degradation = ((baseline - worst) / baseline) * 100
                degradations[depth_name] = degradation

        if degradations:
            bars = ax.bar(degradations.keys(), degradations.values(),
                         color=[self.phases[phase]['color']], alpha=0.7)

            for bar, val in zip(bars, degradations.values()):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.1f}%', ha='center', va='bottom')

            ax.set_ylabel('F1 Score Degradation (%)')
            ax.set_title('Performance Degradation Under Attack')
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'{phase}_attack_robustness_depth.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

        return fig

    def visualize_all_phases_depth_comparison(self,
                                              phase_results: Dict[str, Dict[str, float]],
                                              save: bool = True) -> plt.Figure:
        """
        Create a comprehensive comparison of depth effects across all phases.

        Args:
            phase_results: Dict mapping phase name to dict of depth -> final_f1
            save: Whether to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Depth Impact Across All FL Phases',
                    fontsize=18, fontweight='bold')

        # 1. Grouped bar chart
        ax = axes[0, 0]
        phases_list = list(phase_results.keys())
        depths_list = list(set().union(*[set(d.keys()) for d in phase_results.values()]))

        x = np.arange(len(phases_list))
        width = 0.2

        for i, depth in enumerate(depths_list):
            values = [phase_results[p].get(depth, 0) for p in phases_list]
            ax.bar(x + i * width, values, width, label=depth)

        ax.set_xticks(x + width * (len(depths_list) - 1) / 2)
        ax.set_xticklabels(phases_list, rotation=45, ha='right')
        ax.set_ylabel('Final F1 Score')
        ax.set_title('F1 Score by Phase and Depth')
        ax.legend(title='Depth')
        ax.grid(True, alpha=0.3, axis='y')

        # 2. Heatmap
        ax = axes[0, 1]
        heatmap_data = np.zeros((len(depths_list), len(phases_list)))
        for i, depth in enumerate(depths_list):
            for j, phase in enumerate(phases_list):
                heatmap_data[i, j] = phase_results[phase].get(depth, 0)

        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   xticklabels=phases_list, yticklabels=depths_list, ax=ax)
        ax.set_title('Performance Heatmap (Depth × Phase)')

        # 3. Radar chart
        ax = axes[1, 0]
        angles = np.linspace(0, 2 * np.pi, len(phases_list), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))

        for depth in depths_list:
            values = [phase_results[p].get(depth, 0) for p in phases_list]
            values = np.concatenate((values, [values[0]]))
            ax.plot(angles, values, 'o-', linewidth=2, label=depth, markersize=8)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(phases_list)
        ax.set_ylabel('F1 Score')
        ax.set_title('Depth Performance Radar')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        ax.grid(True)

        # 4. Depth sensitivity analysis
        ax = axes[1, 1]
        sensitivities = {}
        for phase in phases_list:
            scores = list(phase_results[phase].values())
            if scores:
                sensitivity = np.std(scores) / np.mean(scores) * 100
                sensitivities[phase] = sensitivity

        if sensitivities:
            colors = [self.phases.get(p, {}).get('color', '#333') for p in sensitivities.keys()]
            bars = ax.bar(sensitivities.keys(), sensitivities.values(), color=colors, alpha=0.7)

            for bar, val in zip(bars, sensitivities.values()):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.2f}%', ha='center', va='bottom')

            ax.set_ylabel('Coefficient of Variation (%)')
            ax.set_title('Depth Sensitivity by Phase')
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / 'all_phases_depth_comparison.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

        return fig

    def visualize_parameter_efficiency(self,
                                       models: Dict[str, nn.Module],
                                       performance: Dict[str, float],
                                       phase: str,
                                       save: bool = True) -> plt.Figure:
        """
        Visualize parameter efficiency (performance per parameter).

        Args:
            models: Dict mapping depth name to model
            performance: Dict mapping depth name to final F1 score
            phase: Current phase name
            save: Whether to save the figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'Parameter Efficiency Analysis - {self.phases[phase]["name"]}',
                    fontsize=16, fontweight='bold')

        # Calculate metrics
        depth_metrics = {}
        for depth_name, model in models.items():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f1_score = performance.get(depth_name, 0)

            depth_metrics[depth_name] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'f1_score': f1_score,
                'efficiency': f1_score / (total_params / 1000) if total_params > 0 else 0
            }

        # 1. Parameter count comparison
        ax = axes[0]
        depths = list(depth_metrics.keys())
        params = [depth_metrics[d]['total_params'] for d in depths]

        bars = ax.bar(depths, params, color=self.phases[phase]['color'], alpha=0.7)
        for bar, val in zip(bars, params):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:,}', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Total Parameters')
        ax.set_title('Model Size by Depth')
        ax.tick_params(axis='x', rotation=45)

        # 2. Performance vs Parameters scatter
        ax = axes[1]
        for depth_name, metrics in depth_metrics.items():
            ax.scatter(metrics['total_params'], metrics['f1_score'],
                      s=150, label=depth_name, alpha=0.8)
            ax.annotate(depth_name,
                       (metrics['total_params'], metrics['f1_score']),
                       xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('Total Parameters')
        ax.set_ylabel('F1 Score')
        ax.set_title('Performance vs Model Size')
        ax.grid(True, alpha=0.3)

        # 3. Efficiency score
        ax = axes[2]
        efficiencies = {d: depth_metrics[d]['efficiency'] for d in depths}

        bars = ax.bar(efficiencies.keys(), efficiencies.values(),
                     color=self.phases[phase]['color'], alpha=0.7)
        for bar, val in zip(bars, efficiencies.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('F1 Score per 1K Parameters')
        ax.set_title('Parameter Efficiency')
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'{phase}_parameter_efficiency.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

        return fig

    def generate_phase_report(self, phase: str,
                              results: Dict[str, Any],
                              save: bool = True) -> str:
        """
        Generate a comprehensive text report for a phase's depth analysis.

        Args:
            phase: Phase name
            results: Dictionary containing all results for this phase
            save: Whether to save the report
        """
        report = []
        report.append(f"=" * 70)
        report.append(f"DEPTH ANALYSIS REPORT - {self.phases[phase]['name'].upper()}")
        report.append(f"=" * 70)
        report.append("")

        # Summary statistics
        if 'depth_performance' in results:
            report.append("PERFORMANCE BY DEPTH:")
            report.append("-" * 40)
            for depth, perf in results['depth_performance'].items():
                report.append(f"  {depth:15} | F1: {perf:.4f}")
            report.append("")

            # Best depth
            best_depth = max(results['depth_performance'].items(), key=lambda x: x[1])
            report.append(f"BEST PERFORMING DEPTH: {best_depth[0]} (F1: {best_depth[1]:.4f})")
            report.append("")

        # Convergence analysis
        if 'convergence' in results:
            report.append("CONVERGENCE ANALYSIS:")
            report.append("-" * 40)
            for depth, conv_data in results['convergence'].items():
                report.append(f"  {depth}:")
                report.append(f"    - Converged: {conv_data.get('converged', False)}")
                report.append(f"    - Convergence Round: {conv_data.get('round', 'N/A')}")
                report.append(f"    - Final F1: {conv_data.get('final_f1', 0):.4f}")
            report.append("")

        # Parameter efficiency
        if 'efficiency' in results:
            report.append("PARAMETER EFFICIENCY:")
            report.append("-" * 40)
            for depth, eff_data in results['efficiency'].items():
                report.append(f"  {depth}:")
                report.append(f"    - Total Params: {eff_data.get('params', 0):,}")
                report.append(f"    - Efficiency: {eff_data.get('score', 0):.4f} F1/1K params")
            report.append("")

        report.append("=" * 70)
        report_text = "\n".join(report)

        if save:
            filepath = self.output_dir / f'{phase}_depth_report.txt'
            with open(filepath, 'w') as f:
                f.write(report_text)
            print(f"Saved report: {filepath}")

        return report_text


def create_sample_visualizations():
    """Create sample depth visualizations with synthetic data for demonstration."""
    print("Creating sample depth visualizations...")

    visualizer = DepthVisualizer()

    # Sample input dimension (e.g., UNSW-NB15 features)
    input_dim = 42

    # Create models with different depths
    models = {}
    for depth_name, hidden_dims in visualizer.depth_configs.items():
        models[depth_name] = visualizer.create_model_with_depth(input_dim, hidden_dims)

    # Generate sample data
    batch_size = 64
    X_sample = torch.randn(batch_size, input_dim)
    y_sample = torch.randint(0, 2, (batch_size,)).float()

    # Create visualizations for each phase
    phases_to_visualize = ['baseline', 'privacy', 'aggregators', 'attacks', 'full']

    for phase in phases_to_visualize:
        print(f"\nGenerating visualizations for {phase} phase...")

        # 1. Weight distributions
        visualizer.visualize_layer_weights_distribution(models, phase)

        # 2. Gradient flow
        visualizer.visualize_gradient_flow(models, X_sample, y_sample, phase)

        # 3. Activation statistics
        visualizer.visualize_activation_statistics(models, X_sample, phase)

        # 4. Convergence comparison (with synthetic data)
        results_by_depth = {}
        for depth_name in visualizer.depth_configs.keys():
            results_by_depth[depth_name] = []
            base_f1 = 0.5 + np.random.uniform(0, 0.1)
            for round_num in range(1, 51):
                # Simulate convergence (deeper models converge slower but reach higher F1)
                depth_factor = len(visualizer.depth_configs[depth_name])
                f1 = base_f1 + (0.95 - base_f1) * (1 - np.exp(-round_num / (10 * depth_factor)))
                f1 += np.random.uniform(-0.02, 0.02)
                f1 = min(1.0, max(0.0, f1))

                results_by_depth[depth_name].append({
                    'round': round_num,
                    'f1': f1,
                    'loss': 1.0 - f1 + np.random.uniform(0, 0.1),
                    'accuracy': f1 - 0.05 + np.random.uniform(0, 0.1),
                    'precision': f1 + np.random.uniform(-0.05, 0.05),
                    'recall': f1 + np.random.uniform(-0.05, 0.05),
                    'auc_roc': f1 + 0.02 + np.random.uniform(-0.02, 0.02)
                })

        visualizer.visualize_depth_convergence_comparison(results_by_depth, phase)

        # 5. Parameter efficiency
        performance = {}
        for depth_name in visualizer.depth_configs.keys():
            performance[depth_name] = results_by_depth[depth_name][-1]['f1']

        visualizer.visualize_parameter_efficiency(models, performance, phase)

        # 6. Phase-specific visualizations
        if phase == 'privacy':
            # Privacy-depth tradeoff
            privacy_results = {}
            for depth_name in visualizer.depth_configs.keys():
                privacy_results[depth_name] = {
                    0.5: 0.6 + np.random.uniform(0, 0.1),
                    1.0: 0.7 + np.random.uniform(0, 0.1),
                    5.0: 0.85 + np.random.uniform(0, 0.05),
                    10.0: 0.9 + np.random.uniform(0, 0.05)
                }
            visualizer.visualize_privacy_depth_tradeoff(privacy_results, phase)

        elif phase == 'attacks':
            # Attack robustness
            attack_results = {}
            for depth_name in visualizer.depth_configs.keys():
                depth_factor = len(visualizer.depth_configs[depth_name])
                attack_results[depth_name] = {
                    0.0: 0.9 + np.random.uniform(0, 0.05),
                    0.1: 0.85 - 0.02 * depth_factor + np.random.uniform(0, 0.05),
                    0.2: 0.75 - 0.03 * depth_factor + np.random.uniform(0, 0.05),
                    0.3: 0.65 - 0.04 * depth_factor + np.random.uniform(0, 0.05)
                }
            visualizer.visualize_attack_robustness_by_depth(attack_results, phase)

        # Generate report
        results = {
            'depth_performance': performance,
            'convergence': {d: {'converged': True, 'round': 30 + len(visualizer.depth_configs[d]) * 5,
                               'final_f1': performance[d]} for d in performance},
            'efficiency': {d: {'params': sum(p.numel() for p in models[d].parameters()),
                              'score': performance[d] / (sum(p.numel() for p in models[d].parameters()) / 1000)}
                          for d in performance}
        }
        visualizer.generate_phase_report(phase, results)

    # Create overall comparison
    print("\nGenerating cross-phase comparison...")
    phase_results = {}
    for phase in phases_to_visualize:
        phase_results[phase] = {}
        for depth_name in visualizer.depth_configs.keys():
            phase_results[phase][depth_name] = 0.7 + np.random.uniform(0, 0.2)

    visualizer.visualize_all_phases_depth_comparison(phase_results)

    print(f"\nAll visualizations saved to: {visualizer.output_dir}")
    print(f"Total files generated: {len(list(visualizer.output_dir.glob('*.png'))) + len(list(visualizer.output_dir.glob('*.txt')))}")


if __name__ == "__main__":
    create_sample_visualizations()
