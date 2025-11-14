"""
Statistical Analysis Module
============================
Perform rigorous statistical analysis on experiment results.

Features:
- Mean, std, median, CI computation
- Paired t-tests with p-values
- Effect sizes (Cohen's d)
- Bonferroni correction for multiple comparisons
- Statistical significance testing
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from collections import defaultdict


def load_phase_results(results_dir: Path) -> Dict[str, List[Dict]]:
    """Load all results from a phase directory

    Args:
        results_dir: Path to phase results directory

    Returns:
        Dictionary mapping config keys to list of run results
    """
    results = defaultdict(list)

    for json_file in results_dir.glob("*.json"):
        if json_file.name.startswith("summary") or json_file.name.startswith("analysis"):
            continue

        with open(json_file, 'r') as f:
            data = json.load(f)

        # Create config key
        config = data['config']
        key = _config_to_key(config)

        results[key].append(data['summary'])

    return dict(results)


def _config_to_key(config: Dict) -> str:
    """Convert config dict to unique key"""
    eps_str = f"eps_{config['epsilon']}" if config['epsilon'] != float('inf') else "eps_inf"
    agg_str = config['aggregator']
    attack_str = f"{config['attack_type']}_r{config['attack_ratio']}" if config['attack_ratio'] > 0 else "clean"
    return f"{eps_str}_{agg_str}_{attack_str}"


def compute_statistics(values: List[float]) -> Dict:
    """Compute comprehensive statistics for a list of values

    Args:
        values: List of numeric values

    Returns:
        Dictionary with statistical measures
    """
    if not values:
        return {}

    values = np.array(values)

    # Confidence interval (95%)
    ci_95 = stats.t.interval(
        confidence=0.95,
        df=len(values) - 1,
        loc=np.mean(values),
        scale=stats.sem(values)
    ) if len(values) > 1 else (values[0], values[0])

    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        'median': float(np.median(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'ci_95_lower': float(ci_95[0]),
        'ci_95_upper': float(ci_95[1]),
        'n': len(values),
    }


def paired_t_test(group1: List[float], group2: List[float]) -> Dict:
    """Perform paired t-test between two groups

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Dictionary with test results
    """
    if len(group1) != len(group2) or len(group1) < 2:
        return {
            'statistic': None,
            'p_value': None,
            'significant': False,
            'effect_size': None,
        }

    # Paired t-test
    statistic, p_value = stats.ttest_rel(group1, group2)

    # Cohen's d (effect size)
    diff = np.array(group1) - np.array(group2)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'cohens_d': float(cohens_d),
        'effect_size_interpretation': _interpret_effect_size(abs(cohens_d)),
    }


def independent_t_test(group1: List[float], group2: List[float]) -> Dict:
    """Perform independent t-test between two groups

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Dictionary with test results
    """
    if len(group1) < 2 or len(group2) < 2:
        return {
            'statistic': None,
            'p_value': None,
            'significant': False,
            'effect_size': None,
        }

    # Independent t-test
    statistic, p_value = stats.ttest_ind(group1, group2)

    # Cohen's d
    pooled_std = np.sqrt(
        ((len(group1) - 1) * np.var(group1, ddof=1) +
         (len(group2) - 1) * np.var(group2, ddof=1)) /
        (len(group1) + len(group2) - 2)
    )

    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0.0

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'cohens_d': float(cohens_d),
        'effect_size_interpretation': _interpret_effect_size(abs(cohens_d)),
    }


def _interpret_effect_size(cohens_d: float) -> str:
    """Interpret Cohen's d effect size"""
    if cohens_d < 0.2:
        return "negligible"
    elif cohens_d < 0.5:
        return "small"
    elif cohens_d < 0.8:
        return "medium"
    else:
        return "large"


def bonferroni_correction(p_values: List[float]) -> List[float]:
    """Apply Bonferroni correction for multiple comparisons

    Args:
        p_values: List of p-values

    Returns:
        List of corrected p-values
    """
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def analyze_phase_results(results_dir: Path, metric_name: str = 'final_f1') -> Dict:
    """Perform comprehensive statistical analysis on phase results

    Args:
        results_dir: Path to phase results directory
        metric_name: Metric to analyze (default: final_f1)

    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*80}")
    print(f"STATISTICAL ANALYSIS: {results_dir.name}")
    print(f"Metric: {metric_name}")
    print(f"{'='*80}\n")

    # Load results
    results = load_phase_results(results_dir)

    if not results:
        print("⚠️  No results found")
        return {}

    print(f"Loaded {len(results)} configurations\n")

    # Compute statistics for each config
    config_stats = {}

    print("Configuration Statistics:")
    print("-" * 80)

    for config_key, runs in sorted(results.items()):
        values = [run[metric_name] for run in runs if metric_name in run]

        if not values:
            continue

        stats_dict = compute_statistics(values)
        config_stats[config_key] = {
            'values': values,
            'stats': stats_dict,
        }

        print(f"\n{config_key}:")
        print(f"  Mean: {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}")
        print(f"  Median: {stats_dict['median']:.4f}")
        print(f"  Range: [{stats_dict['min']:.4f}, {stats_dict['max']:.4f}]")
        print(f"  95% CI: [{stats_dict['ci_95_lower']:.4f}, {stats_dict['ci_95_upper']:.4f}]")
        print(f"  N: {stats_dict['n']}")

    # Pairwise comparisons
    print(f"\n{'='*80}")
    print("PAIRWISE COMPARISONS")
    print(f"{'='*80}\n")

    comparisons = []
    config_keys = list(config_stats.keys())

    for i in range(len(config_keys)):
        for j in range(i + 1, len(config_keys)):
            key1, key2 = config_keys[i], config_keys[j]

            values1 = config_stats[key1]['values']
            values2 = config_stats[key2]['values']

            # Use independent t-test (not paired) since runs may differ
            test_result = independent_t_test(values1, values2)

            if test_result['p_value'] is not None:
                comparisons.append({
                    'config1': key1,
                    'config2': key2,
                    'mean1': config_stats[key1]['stats']['mean'],
                    'mean2': config_stats[key2]['stats']['mean'],
                    'mean_diff': config_stats[key1]['stats']['mean'] - config_stats[key2]['stats']['mean'],
                    **test_result
                })

    # Sort by p-value
    comparisons.sort(key=lambda x: x['p_value'] if x['p_value'] is not None else 1.0)

    # Apply Bonferroni correction
    p_values = [c['p_value'] for c in comparisons]
    corrected_p_values = bonferroni_correction(p_values)

    for i, comp in enumerate(comparisons):
        comp['p_value_corrected'] = corrected_p_values[i]
        comp['significant_corrected'] = corrected_p_values[i] < 0.05

    # Print top comparisons
    print("Top 10 Most Significant Comparisons (Bonferroni corrected):")
    print("-" * 80)

    for i, comp in enumerate(comparisons[:10]):
        print(f"\n{i+1}. {comp['config1']}")
        print(f"   vs")
        print(f"   {comp['config2']}")
        print(f"   Mean diff: {comp['mean_diff']:.4f} ({comp['mean1']:.4f} vs {comp['mean2']:.4f})")
        print(f"   p-value: {comp['p_value']:.4e} (corrected: {comp['p_value_corrected']:.4e})")
        print(f"   Cohen's d: {comp['cohens_d']:.3f} ({comp['effect_size_interpretation']})")
        print(f"   Significant: {'✓ YES' if comp['significant_corrected'] else '✗ NO'}")

    # Find best configuration
    print(f"\n{'='*80}")
    print("BEST CONFIGURATIONS")
    print(f"{'='*80}\n")

    best_configs = sorted(
        config_stats.items(),
        key=lambda x: x[1]['stats']['mean'],
        reverse=True
    )

    print("Top 5 Configurations by Mean Performance:")
    for i, (config_key, data) in enumerate(best_configs[:5]):
        stats_dict = data['stats']
        print(f"\n{i+1}. {config_key}")
        print(f"   Mean: {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}")
        print(f"   95% CI: [{stats_dict['ci_95_lower']:.4f}, {stats_dict['ci_95_upper']:.4f}]")

    # Save analysis results
    analysis = {
        'metric': metric_name,
        'config_statistics': {
            key: data['stats'] for key, data in config_stats.items()
        },
        'pairwise_comparisons': comparisons,
        'best_configurations': [
            {'config': key, **data['stats']}
            for key, data in best_configs
        ],
    }

    analysis_file = results_dir / 'analysis' / f'statistical_analysis_{metric_name}.json'
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)  # Convert non-serializable types to string

    print(f"\n✓ Analysis saved to: {analysis_file}")

    return analysis


def create_summary_table(results_dir: Path, metrics: List[str] = None) -> pd.DataFrame:
    """Create summary table for all configurations

    Args:
        results_dir: Path to phase results directory
        metrics: List of metrics to include (default: common metrics)

    Returns:
        Pandas DataFrame with summary statistics
    """
    if metrics is None:
        metrics = ['final_f1', 'final_accuracy', 'final_auc_roc', 'total_time', 'convergence_round']

    results = load_phase_results(results_dir)

    rows = []
    for config_key, runs in sorted(results.items()):
        row = {'config': config_key}

        for metric in metrics:
            values = [run.get(metric, np.nan) for run in runs]
            values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]

            if values:
                stats_dict = compute_statistics(values)
                row[f'{metric}_mean'] = stats_dict['mean']
                row[f'{metric}_std'] = stats_dict['std']
                row[f'{metric}_ci_lower'] = stats_dict['ci_95_lower']
                row[f'{metric}_ci_upper'] = stats_dict['ci_95_upper']

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save to CSV
    csv_file = results_dir / 'analysis' / 'summary_table.csv'
    df.to_csv(csv_file, index=False)

    print(f"✓ Summary table saved to: {csv_file}")

    return df


def compare_aggregators(results_dir: Path, metric: str = 'final_f1') -> Dict:
    """Compare different aggregation strategies

    Args:
        results_dir: Path to phase results directory
        metric: Metric to compare

    Returns:
        Comparison results
    """
    print(f"\n{'='*80}")
    print(f"AGGREGATOR COMPARISON")
    print(f"{'='*80}\n")

    results = load_phase_results(results_dir)

    # Group by aggregator
    aggregator_results = defaultdict(list)

    for config_key, runs in results.items():
        # Extract aggregator from config key
        for agg in ['fedavg', 'trimmed_mean', 'median']:
            if agg in config_key:
                values = [run[metric] for run in runs if metric in run]
                aggregator_results[agg].extend(values)
                break

    # Compute statistics
    agg_stats = {}
    for agg, values in aggregator_results.items():
        agg_stats[agg] = compute_statistics(values)

    # Print statistics
    print("Aggregator Statistics:")
    print("-" * 80)
    for agg, stats_dict in sorted(agg_stats.items()):
        print(f"\n{agg.upper()}:")
        print(f"  Mean: {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}")
        print(f"  Median: {stats_dict['median']:.4f}")
        print(f"  95% CI: [{stats_dict['ci_95_lower']:.4f}, {stats_dict['ci_95_upper']:.4f}]")
        print(f"  N: {stats_dict['n']}")

    # Pairwise comparisons
    print(f"\nPairwise Comparisons:")
    print("-" * 80)

    agg_names = list(aggregator_results.keys())
    for i in range(len(agg_names)):
        for j in range(i + 1, len(agg_names)):
            agg1, agg2 = agg_names[i], agg_names[j]
            test_result = independent_t_test(
                aggregator_results[agg1],
                aggregator_results[agg2]
            )

            print(f"\n{agg1.upper()} vs {agg2.upper()}:")
            print(f"  Mean diff: {agg_stats[agg1]['mean'] - agg_stats[agg2]['mean']:.4f}")
            print(f"  p-value: {test_result['p_value']:.4e}")
            print(f"  Cohen's d: {test_result['cohens_d']:.3f} ({test_result['effect_size_interpretation']})")
            print(f"  Significant: {'✓ YES' if test_result['significant'] else '✗ NO'}")

    return {
        'statistics': agg_stats,
        'results': dict(aggregator_results),
    }
