# FL-NIDS Comprehensive Experiment System

## Overview

This is a complete rewrite of your FL-NIDS experiment code with:

✅ **Modular architecture** - Separate files for config, metrics, runner, analysis, main
✅ **Phased execution** - Run experiments in phases (baseline, privacy, aggregators, attacks, full)
✅ **Statistical rigor** - 5 runs per config, mean±std, 95% CI, t-tests, effect sizes
✅ **Checkpoint/resume** - Automatically saves and resumes experiments
✅ **Detailed logging** - Progress printed for every round, every experiment
✅ **Comprehensive metrics** - Per-round tracking, convergence detection, early stopping

## File Structure

```
FL_research/
├── config.py           # Experiment configurations (CHANGE PHASE HERE)
├── metrics.py          # Metrics tracking and computation
├── runner.py           # Single experiment executor (Flower + Opacus)
├── analysis.py         # Statistical analysis functions
├── main.py             # Main orchestrator (RUN THIS)
├── nwdaf_security_analytics_.py  # Original pilot code (backup)
└── results/
    ├── phase1_baseline/
    │   ├── eps_inf_fedavg_clean_run0.json
    │   ├── eps_inf_fedavg_clean_run1.json
    │   ├── ...
    │   └── analysis/
    │       ├── statistical_analysis_final_f1.json
    │       └── summary_table.csv
    ├── phase2_privacy/
    ├── phase3_aggregators/
    ├── phase4_attacks/
    └── phase5_full/
```

## Quick Start

### 1. Run Baseline Experiments

```bash
# Edit config.py and set:
PHASE = "baseline"

# Run experiments (Google Colab):
!python main.py
```

This will:
- Run 3 configs (fedavg, trimmed_mean, median) × 5 runs = 15 experiments
- Save results incrementally
- Compute statistical analysis
- Print detailed progress

### 2. Run Privacy Analysis

```bash
# Edit config.py and set:
PHASE = "privacy"

# Run experiments:
!python main.py
```

This will:
- Test 7 epsilon values × 5 runs = 35 experiments
- Analyze privacy-utility tradeoff
- Compare with baseline

### 3. Run Other Phases

```bash
PHASE = "aggregators"  # Test aggregator robustness under attacks
PHASE = "attacks"      # Comprehensive attack scenarios with DP
PHASE = "full"         # Complete grid (for final paper)
```

## Configuration Options

### Phase Definitions (config.py)

**Baseline** - No DP, no attacks, test all aggregators
```python
PHASE = "baseline"
# 3 configs × 5 runs = 15 experiments
```

**Privacy** - Vary epsilon values, FedAvg only
```python
PHASE = "privacy"
# 7 epsilon values × 5 runs = 35 experiments
```

**Aggregators** - Test robustness under attacks
```python
PHASE = "aggregators"
# 3 aggregators × 2 attacks × 4 ratios × 5 runs = 120 experiments
```

**Attacks** - Comprehensive attack analysis with DP
```python
PHASE = "attacks"
# 3 epsilon × 3 aggregators × 3 ratios × 5 runs = 135 experiments
```

**Full** - Complete experimental grid
```python
PHASE = "full"
# 4 epsilon × 3 aggregators × 2 attacks × 4 ratios × 5 runs = 480 experiments
```

### Core Parameters (config.py)

```python
# Experiment settings
NUM_RUNS = 5              # Runs per config (for statistics)
NUM_ROUNDS = 100          # Max FL rounds
NUM_CLIENTS = 10          # Number of clients

# Early stopping
CONVERGENCE_THRESHOLD = 0.90  # F1 score for convergence
EARLY_STOP_PATIENCE = 10      # Rounds without improvement

# Training
LOCAL_EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# Model
HIDDEN_DIMS = [128, 64, 32]
DROPOUT_RATE = 0.3

# Differential Privacy
TARGET_DELTA = 1e-5
MAX_GRAD_NORM = 1.0
```

## Expected Output

### During Execution

```
================================================================================
PHASE: Privacy Analysis
================================================================================
Description: Analyze privacy-utility tradeoff across epsilon values
Total configs: 7
Runs per config: 5
Total experiments: 35
Max rounds: 100
...

================================================================================
Configuration 1/7
================================================================================
  Epsilon: 0.1
  Aggregator: fedavg
  Attack: none (0%)
================================================================================

[12:45:30] Starting Experiment:
  Config: ε=0.1, Agg=fedavg, Attack=none (0%)
  Run: 1/5
  Seed: 42

[12:45:31] Partitioning data across 10 clients...
[12:45:32] Strategy: fedavg
[12:45:33] Starting FL training...

[12:45:48] Round 10/100: F1=0.8234, Loss=0.3456, Acc=0.8567, Time=15.3s
[12:46:03] Round 20/100: F1=0.8567, Loss=0.2891, Acc=0.8823, Time=30.1s
...
[12:47:12] ✓ Converged at round 47 (F1=0.9012)
...
[12:49:45] ✓ Experiment complete
  Final F1: 0.9012
  Best F1: 0.9034 (round 52)
  Total time: 245.3s
  Converged: Round 47

[12:49:45] ✓ Saved: eps_0.1_fedavg_clean_run0.json

================================================================================
PROGRESS UPDATE
================================================================================
  Completed: 1/35 (2.9%)
  Elapsed time: 0.07 hours (4.1 minutes)
  Avg time per experiment: 4.1 minutes
  Est. remaining time: 2.32 hours (139.4 minutes)
================================================================================
```

### After Completion

```
================================================================================
EXPERIMENTS COMPLETE!
================================================================================
Total time: 2.45 hours (147.0 minutes)
Average time per experiment: 4.2 minutes
Results saved to: ./results/phase2_privacy

================================================================================
STEP 4: STATISTICAL ANALYSIS
================================================================================

Configuration Statistics:
--------------------------------------------------------------------------------

eps_0.1_fedavg_clean:
  Mean: 0.8234 ± 0.0123
  Median: 0.8245
  Range: [0.8091, 0.8367]
  95% CI: [0.8102, 0.8366]
  N: 5

eps_0.2_fedavg_clean:
  Mean: 0.8567 ± 0.0098
  Median: 0.8573
  Range: [0.8445, 0.8678]
  95% CI: [0.8462, 0.8672]
  N: 5

...

================================================================================
PAIRWISE COMPARISONS
================================================================================

Top 10 Most Significant Comparisons (Bonferroni corrected):

1. eps_0.1_fedavg_clean
   vs
   eps_inf_fedavg_clean
   Mean diff: -0.0567 (0.8234 vs 0.8801)
   p-value: 1.234e-04 (corrected: 2.345e-03)
   Cohen's d: 2.345 (large)
   Significant: ✓ YES

...

================================================================================
BEST CONFIGURATIONS
================================================================================

Top 5 Configurations by Mean Performance:

1. eps_inf_fedavg_clean
   Mean: 0.8801 ± 0.0045
   95% CI: [0.8751, 0.8851]

2. eps_5.0_fedavg_clean
   Mean: 0.8734 ± 0.0067
   95% CI: [0.8661, 0.8807]

...

✓ Analysis saved to: ./results/phase2_privacy/analysis/statistical_analysis_final_f1.json
✓ Summary table saved to: ./results/phase2_privacy/analysis/summary_table.csv
```

## Metrics Tracked

### Per Round
- `loss` - Training loss
- `accuracy`, `precision`, `recall`, `f1` - Classification metrics
- `auc_roc` - ROC AUC score
- `auc_pr` - Precision-Recall AUC
- `latency` - Round execution time
- `epsilon_consumed` - Privacy budget used (if DP enabled)

### Per Experiment
- `final_*` - Metrics at last round
- `best_*` - Best metrics across all rounds
- `avg_*` - Average metrics
- `convergence_round` - Round where F1 >= 0.90
- `early_stop_round` - Round where early stopping triggered
- `total_time` - Total experiment time

### Statistical Analysis
- `mean`, `std`, `median`, `min`, `max`
- `ci_95_lower`, `ci_95_upper` - 95% confidence interval
- `p_value`, `p_value_corrected` - Statistical significance
- `cohens_d` - Effect size

## Checkpoint & Resume

The system automatically saves after each experiment. If interrupted:

```bash
# Just run again:
!python main.py

# Output will show:
# Total experiments: 35
# Completed: 12
# Remaining: 23
# ↻ Resuming from checkpoint (12 experiments done)
```

Already completed experiments are skipped.

## Troubleshooting

### Out of Memory (OOM)

**Solution 1**: Reduce clients or batch size
```python
NUM_CLIENTS = 5      # Instead of 10
BATCH_SIZE = 64      # Instead of 128
```

**Solution 2**: Use sampling
```python
# In get_phase_config():
'use_sample': True,
'sample_fraction': 0.3,  # Use 30% of data
```

### Experiments Too Slow

**Solution 1**: Reduce rounds
```python
NUM_ROUNDS = 50      # Instead of 100
```

**Solution 2**: Reduce runs per config
```python
NUM_RUNS = 3         # Instead of 5
```

**Solution 3**: Test fewer configs
```python
# In get_phase_config(), reduce epsilon_values:
'epsilon_values': [1.0, 5.0, float('inf')]  # Instead of 7 values
```

### Google Colab Disconnects

The checkpoint system handles this automatically. Just reconnect and run again:

```bash
!python main.py
```

### Data Not Found

Update the path in `config.py`:

```python
DATA_PATH = '/content/drive/MyDrive/IDSDatasets/UNSW 15'
```

## Results Structure

Each experiment saves a JSON file:

```json
{
  "config": {
    "epsilon": 1.0,
    "aggregator": "fedavg",
    "attack_type": "none",
    "attack_ratio": 0.0
  },
  "summary": {
    "final_f1": 0.8801,
    "best_f1": 0.8834,
    "convergence_round": 47,
    "total_time": 245.3,
    ...
  },
  "round_metrics": [
    {"round": 1, "f1": 0.7234, "loss": 0.4567, ...},
    {"round": 2, "f1": 0.7456, "loss": 0.4123, ...},
    ...
  ]
}
```

## Customization

### Add New Phase

Edit `config.py` in `get_phase_config()`:

```python
elif cls.PHASE == "my_custom_phase":
    return {
        'name': 'My Custom Experiments',
        'description': 'Test specific scenarios',
        'configs': [
            {
                'epsilon': 1.0,
                'aggregator': 'fedavg',
                'attack_type': 'none',
                'attack_ratio': 0.0,
            },
            # ... more configs
        ],
        'num_rounds': 100,
        'use_sample': False,
        'sample_fraction': 1.0,
    }
```

### Add New Metric

Edit `metrics.py` in `compute_comprehensive_metrics()`:

```python
# Add your custom metric
metrics['my_custom_metric'] = your_calculation(y_true, y_pred)
```

### Add New Aggregator

Edit `runner.py`:

```python
class MyCustomStrategy(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Your aggregation logic
        ...
```

Then in `run_single_experiment()`:

```python
if experiment_config['aggregator'] == 'my_custom':
    strategy = MyCustomStrategy(**strategy_params)
```

## Comparison with Original Code

| Feature | Original | New System |
|---------|----------|------------|
| Structure | Single file (1128 lines) | Modular (5 files) |
| Execution | All configs at once | Phased |
| Runs per config | 1 | 5 (configurable) |
| Statistics | None | Mean, std, CI, t-tests |
| Checkpointing | No | Yes (automatic) |
| Progress tracking | Minimal | Detailed per-round |
| Early stopping | No | Yes |
| Convergence detection | No | Yes |
| Resume capability | No | Yes |
| Statistical analysis | No | Comprehensive |

## What's Preserved

✅ All existing Flower + Opacus integration
✅ Same model architecture
✅ Same data preprocessing
✅ Same aggregation strategies (FedAvg, TrimmedMean, Median)
✅ Same attack implementations (label_flip)
✅ Same privacy mechanisms (Opacus DP)

## Next Steps

1. **Start with baseline**: Set `PHASE = "baseline"` and run
2. **Check results**: Look at `results/phase1_baseline/analysis/`
3. **Move to privacy**: Set `PHASE = "privacy"` and run
4. **Continue phases**: aggregators → attacks → full
5. **Use analysis**: Statistical tests in `analysis/*.json`

## Support

If you encounter issues:

1. Check the detailed error messages
2. Verify `DATA_PATH` in config.py
3. Try reducing `NUM_CLIENTS` or `BATCH_SIZE` if OOM
4. Use `use_sample=True` for faster testing

Good luck with your experiments! 🚀
