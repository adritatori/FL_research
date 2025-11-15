"""
Phase 2 Results Visualization
==============================
Comprehensive visualization of Phase 2 experiment results.

Creates multiple plots:
- Performance comparison across configurations
- Box plots showing distributions
- Privacy vs Performance trade-offs
- Attack resistance analysis
- Aggregator comparison
- Statistical significance heatmaps

Usage:
    python visualize_phase2.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def load_phase_results(results_dir):
    """Load all experiment results from JSON files"""
    results = defaultdict(list)

    for json_file in results_dir.glob("*.json"):
        if json_file.name.startswith("summary") or json_file.name.startswith("analysis"):
            continue

        with open(json_file, 'r') as f:
            data = json.load(f)

        config = data['config']
        key = create_config_key(config)
        results[key].append({
            'config': config,
            'summary': data['summary'],
            'history': data.get('history', {})
        })

    return dict(results)

def create_config_key(config):
    """Create readable key from config"""
    eps = f"ε={config['epsilon']:.1f}" if config['epsilon'] != float('inf') else "No DP"
    agg = config['aggregator'].replace('_', ' ').title()

    if config['attack_ratio'] > 0:
        attack = f"{config['attack_type'].title()} {int(config['attack_ratio']*100)}%"
    else:
        attack = "Clean"

    return f"{agg} | {eps} | {attack}"

def plot_performance_comparison(results, metric='final_f1', save_path=None):
    """Bar plot comparing performance across configurations"""
    fig, ax = plt.subplots(figsize=(14, 6))

    configs = []
    means = []
    stds = []

    for config_key, runs in sorted(results.items()):
        values = [run['summary'][metric] for run in runs if metric in run['summary']]
        if values:
            configs.append(config_key)
            means.append(np.mean(values))
            stds.append(np.std(values))

    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')

    # Color bars by performance
    colors = plt.cm.RdYlGn(np.array(means) / max(means))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'Performance Comparison: {metric.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_metric_distributions(results, metrics=['final_f1', 'final_accuracy', 'final_auc_roc'], save_path=None):
    """Box plots showing distributions of multiple metrics"""
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        data_for_plot = []
        labels = []

        for config_key, runs in sorted(results.items()):
            values = [run['summary'].get(metric) for run in runs]
            values = [v for v in values if v is not None]
            if values:
                data_for_plot.append(values)
                labels.append(config_key)

        bp = axes[idx].boxplot(data_for_plot, labels=labels, patch_artist=True)

        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_for_plot)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[idx].set_xlabel('Configuration', fontweight='bold')
        axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
        axes[idx].set_title(f'{metric.replace("_", " ").title()} Distribution', fontweight='bold')
        axes[idx].tick_params(axis='x', rotation=45)
        plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_privacy_performance_tradeoff(results, save_path=None):
    """Scatter plot showing privacy (epsilon) vs performance"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by aggregator
    aggregators = defaultdict(lambda: {'epsilon': [], 'f1': [], 'attack': []})

    for config_key, runs in results.items():
        config = runs[0]['config']
        agg = config['aggregator']
        eps = config['epsilon']

        if eps == float('inf'):
            continue  # Skip non-DP for this plot

        f1_values = [run['summary']['final_f1'] for run in runs if 'final_f1' in run['summary']]
        if f1_values:
            aggregators[agg]['epsilon'].append(eps)
            aggregators[agg]['f1'].append(np.mean(f1_values))
            aggregators[agg]['attack'].append('Attack' if config['attack_ratio'] > 0 else 'Clean')

    # Plot each aggregator
    markers = {'fedavg': 'o', 'trimmed_mean': 's', 'median': '^'}
    colors = {'Clean': 'green', 'Attack': 'red'}

    for agg, data in aggregators.items():
        for i, (eps, f1, attack) in enumerate(zip(data['epsilon'], data['f1'], data['attack'])):
            marker = markers.get(agg, 'o')
            color = colors.get(attack, 'blue')
            label = f"{agg} ({attack})" if i == 0 else ""
            ax.scatter(eps, f1, marker=marker, s=100, c=color, alpha=0.6,
                      edgecolors='black', linewidth=1.5, label=label)

    ax.set_xlabel('Privacy Budget (ε)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Privacy-Performance Trade-off', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.text(0.02, 0.98, 'Lower ε = More Privacy\nHigher F1 = Better Performance',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_aggregator_comparison(results, save_path=None):
    """Compare aggregator performance under different conditions"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Group by aggregator and attack status
    agg_data = defaultdict(lambda: {'clean': [], 'attack': []})

    for config_key, runs in results.items():
        config = runs[0]['config']
        agg = config['aggregator']
        f1_values = [run['summary']['final_f1'] for run in runs if 'final_f1' in run['summary']]

        if f1_values:
            if config['attack_ratio'] > 0:
                agg_data[agg]['attack'].extend(f1_values)
            else:
                agg_data[agg]['clean'].extend(f1_values)

    # Plot 1: Clean data performance
    clean_data = [agg_data[agg]['clean'] for agg in sorted(agg_data.keys()) if agg_data[agg]['clean']]
    clean_labels = [agg.replace('_', ' ').title() for agg in sorted(agg_data.keys()) if agg_data[agg]['clean']]

    if clean_data:
        bp1 = axes[0].boxplot(clean_data, labels=clean_labels, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightgreen')
            patch.set_alpha(0.7)
        axes[0].set_ylabel('F1 Score', fontweight='bold')
        axes[0].set_title('Performance on Clean Data', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: Under attack performance
    attack_data = [agg_data[agg]['attack'] for agg in sorted(agg_data.keys()) if agg_data[agg]['attack']]
    attack_labels = [agg.replace('_', ' ').title() for agg in sorted(agg_data.keys()) if agg_data[agg]['attack']]

    if attack_data:
        bp2 = axes[1].boxplot(attack_data, labels=attack_labels, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightcoral')
            patch.set_alpha(0.7)
        axes[1].set_ylabel('F1 Score', fontweight='bold')
        axes[1].set_title('Performance Under Attack', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_convergence_curves(results, save_path=None):
    """Plot convergence curves for different configurations"""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (config_key, runs), color in zip(sorted(results.items()), colors):
        # Average across runs
        all_histories = []
        max_rounds = 0

        for run in runs:
            if 'history' in run and 'f1_scores' in run['history']:
                f1_history = run['history']['f1_scores']
                all_histories.append(f1_history)
                max_rounds = max(max_rounds, len(f1_history))

        if all_histories:
            # Pad histories to same length
            padded = [hist + [hist[-1]] * (max_rounds - len(hist)) for hist in all_histories]
            mean_history = np.mean(padded, axis=0)
            std_history = np.std(padded, axis=0)

            rounds = np.arange(1, len(mean_history) + 1)
            ax.plot(rounds, mean_history, label=config_key, color=color, linewidth=2)
            ax.fill_between(rounds, mean_history - std_history, mean_history + std_history,
                           alpha=0.2, color=color)

    ax.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Curves', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_metrics_heatmap(results, save_path=None):
    """Heatmap showing all metrics across configurations"""
    metrics = ['final_f1', 'final_accuracy', 'final_precision', 'final_recall',
               'final_auc_roc', 'convergence_round']

    data = []
    labels = []

    for config_key, runs in sorted(results.items()):
        row = []
        for metric in metrics:
            values = [run['summary'].get(metric) for run in runs]
            values = [v for v in values if v is not None]
            if values:
                row.append(np.mean(values))
            else:
                row.append(np.nan)

        if not all(np.isnan(v) for v in row):
            data.append(row)
            labels.append(config_key)

    df = pd.DataFrame(data, columns=[m.replace('_', ' ').title() for m in metrics], index=labels)

    # Normalize each column for better visualization
    df_norm = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x, axis=0)

    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.5)))
    sns.heatmap(df_norm, annot=df, fmt='.3f', cmap='YlGnBu', cbar_kws={'label': 'Normalized Score'},
                linewidths=0.5, ax=ax)
    ax.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
    ax.set_ylabel('Configuration', fontweight='bold')
    ax.set_xlabel('Metric', fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_attack_resistance(results, save_path=None):
    """Analyze attack resistance of different configurations"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by aggregator and privacy level
    grouped = defaultdict(lambda: {'clean': [], 'attack': [], 'config': None})

    for config_key, runs in results.items():
        config = runs[0]['config']

        # Create grouping key (aggregator + privacy)
        eps_str = f"ε={config['epsilon']:.1f}" if config['epsilon'] != float('inf') else "No DP"
        group_key = f"{config['aggregator']}_{eps_str}"

        f1_values = [run['summary']['final_f1'] for run in runs if 'final_f1' in run['summary']]

        if f1_values:
            if config['attack_ratio'] > 0:
                grouped[group_key]['attack'] = f1_values
            else:
                grouped[group_key]['clean'] = f1_values
            grouped[group_key]['config'] = config

    # Calculate performance drop
    labels = []
    clean_means = []
    attack_means = []
    drops = []

    for key, data in sorted(grouped.items()):
        if data['clean'] and data['attack']:
            clean_mean = np.mean(data['clean'])
            attack_mean = np.mean(data['attack'])
            drop = ((clean_mean - attack_mean) / clean_mean) * 100

            labels.append(key.replace('_', '\n'))
            clean_means.append(clean_mean)
            attack_means.append(attack_mean)
            drops.append(drop)

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, clean_means, width, label='Clean', color='lightgreen', edgecolor='black')
    bars2 = ax.bar(x + width/2, attack_means, width, label='Under Attack', color='lightcoral', edgecolor='black')

    # Add performance drop labels
    for i, (clean, attack, drop) in enumerate(zip(clean_means, attack_means, drops)):
        ax.text(i, max(clean, attack) + 0.02, f'-{drop:.1f}%',
                ha='center', fontsize=8, fontweight='bold', color='red')

    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Attack Resistance Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report(results, save_path=None):
    """Create a summary statistics table"""
    fig, ax = plt.subplots(figsize=(14, max(6, len(results) * 0.4)))
    ax.axis('tight')
    ax.axis('off')

    # Prepare data
    table_data = []
    headers = ['Configuration', 'F1 Mean±SD', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC', 'Rounds']

    for config_key, runs in sorted(results.items(),
                                   key=lambda x: np.mean([r['summary'].get('final_f1', 0) for r in x[1]]),
                                   reverse=True):
        row = [config_key[:40] + '...' if len(config_key) > 40 else config_key]

        for metric in ['final_f1', 'final_accuracy', 'final_precision', 'final_recall', 'final_auc_roc']:
            values = [run['summary'].get(metric) for run in runs]
            values = [v for v in values if v is not None]
            if values:
                row.append(f"{np.mean(values):.3f}±{np.std(values):.3f}")
            else:
                row.append('N/A')

        # Convergence rounds
        rounds = [run['summary'].get('convergence_round') for run in runs]
        rounds = [r for r in rounds if r is not None]
        if rounds:
            row.append(f"{np.mean(rounds):.0f}")
        else:
            row.append('N/A')

        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('Phase 2 Results Summary', fontsize=16, fontweight='bold', pad=20)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main visualization function"""
    print("\n" + "="*80)
    print("PHASE 2 RESULTS VISUALIZATION")
    print("="*80 + "\n")

    # Find results directory
    results_dir = Path('results/phase_2')

    if not results_dir.exists():
        print(f"❌ ERROR: Results directory not found: {results_dir}")
        print("\nPlease make sure you have Phase 2 results in the 'results/phase_2' directory")
        return

    # Create output directory for plots
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    print(f"📁 Loading results from: {results_dir}")
    results = load_phase_results(results_dir)

    if not results:
        print("❌ No results found!")
        return

    print(f"✓ Loaded {len(results)} configurations")
    print(f"📊 Generating visualizations...\n")

    # Generate all plots
    print("1/9 Creating performance comparison...")
    plot_performance_comparison(results, metric='final_f1',
                               save_path=plots_dir / '1_performance_comparison.png')

    print("2/9 Creating metric distributions...")
    plot_metric_distributions(results,
                             save_path=plots_dir / '2_metric_distributions.png')

    print("3/9 Creating privacy-performance trade-off...")
    plot_privacy_performance_tradeoff(results,
                                      save_path=plots_dir / '3_privacy_performance.png')

    print("4/9 Creating aggregator comparison...")
    plot_aggregator_comparison(results,
                               save_path=plots_dir / '4_aggregator_comparison.png')

    print("5/9 Creating convergence curves...")
    plot_convergence_curves(results,
                           save_path=plots_dir / '5_convergence_curves.png')

    print("6/9 Creating metrics heatmap...")
    plot_metrics_heatmap(results,
                        save_path=plots_dir / '6_metrics_heatmap.png')

    print("7/9 Creating attack resistance analysis...")
    plot_attack_resistance(results,
                          save_path=plots_dir / '7_attack_resistance.png')

    print("8/9 Creating accuracy comparison...")
    plot_performance_comparison(results, metric='final_accuracy',
                               save_path=plots_dir / '8_accuracy_comparison.png')

    print("9/9 Creating summary report...")
    create_summary_report(results,
                         save_path=plots_dir / '9_summary_report.png')

    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\n📁 Plots saved to: {plots_dir}")
    print("\nGenerated files:")
    for i, name in enumerate([
        '1_performance_comparison.png',
        '2_metric_distributions.png',
        '3_privacy_performance.png',
        '4_aggregator_comparison.png',
        '5_convergence_curves.png',
        '6_metrics_heatmap.png',
        '7_attack_resistance.png',
        '8_accuracy_comparison.png',
        '9_summary_report.png'
    ], 1):
        print(f"  {i}. {name}")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
