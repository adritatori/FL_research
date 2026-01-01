# Fairness Validation Experiment

This experiment validates whether differential privacy (DP) disproportionately affects detection of certain attack types in federated learning for intrusion detection.

## Overview

**Research Question:** Does DP impact detection rates equally across different attack types, or does it introduce fairness issues?

**Approach:** Run federated learning experiments at different privacy levels (ε = ∞, 5.0, 3.0, 1.0) and measure per-attack-type detection rates.

## Experimental Setup

- **Dataset:** UNSW-NB15 (9 attack categories + Normal traffic)
- **Task:** Binary classification (Normal=0, Attack=1)
- **Evaluation:** Per-attack-type detection rates and fairness metrics
- **Configurations:**
  - Privacy levels: ε = ∞ (no DP), 5.0, 3.0, 1.0
  - Aggregator: Median only
  - Clean data (no Byzantine attacks)
  - 3 runs per configuration (seeds: 42, 142, 242)
  - **Total:** 12 experiments

## Architecture

### Model: IntrusionDetectionMLP
```
Input → Linear(input_dim, 64) → GroupNorm(1, 64) → ReLU → Dropout(0.1)
     → Linear(64, 32) → GroupNorm(1, 32) → ReLU → Dropout(0.1)
     → Linear(32, 1) → Output (binary)
```

### Federated Learning (Flower)
- **Clients:** 10
- **Rounds:** 50
- **Local epochs:** 5
- **Batch size:** 256
- **Learning rate:** Adaptive based on ε
  - ε ≤ 1.0: LR = 0.05
  - ε ≤ 5.0: LR = 0.01
  - ε > 5.0: LR = 0.005
- **Client fraction:** 1.0
- **Min fit clients:** 8

### Differential Privacy (Opacus)
- **Target delta:** 1e-5
- **Max grad norm:** 10.0

### Aggregation
- **Strategy:** Coordinate-wise median

## Prerequisites

```bash
pip install torch numpy pandas scikit-learn flwr opacus
```

## Data Requirements

The script expects the following files in `/content/drive/MyDrive/IDSDatasets/UNSW 15/`:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

**Note:** If your data is in a different location, update the `DATA_DIR` constant in `run_fairness_validation.py` (line 95).

Both files must contain:
- Feature columns
- `label` column (0=Normal, 1=Attack)
- `attack_cat` column with values: Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms

## Usage

### Run All Experiments

```bash
cd fairness_experiment
python run_fairness_validation.py
```

This will run all 12 experiments sequentially and save results.

**Estimated runtime:** 6-8 hours on Colab T4 GPU (~30-40 minutes per experiment)

### Run Specific Configuration (Manual)

You can modify the script to run specific configurations by editing the constants:

```python
EPSILON_VALUES = [5.0]  # Run only ε=5.0
SEEDS = [42]            # Run only seed 42
```

## Output Structure

```
fairness_experiment/
├── run_fairness_validation.py
├── README.md
└── results/
    ├── eps_inf_run0.json        # Results for ε=∞, run 0
    ├── eps_inf_run1.json        # Results for ε=∞, run 1
    ├── eps_inf_run2.json        # Results for ε=∞, run 2
    ├── eps_5.0_run0.json        # Results for ε=5.0, run 0
    ├── ...                      # (12 total JSON files)
    ├── summary.csv              # Combined summary of all experiments
    └── models/
        ├── eps_inf_run0.pt      # Model weights for ε=∞, run 0
        ├── eps_inf_run1.pt
        └── ...                  # (12 total model files)
```

## Results Format

### Individual Experiment JSON

Each experiment produces a JSON file with:

```json
{
  "config": {
    "epsilon": 5.0,
    "aggregator": "median",
    "seed": 42,
    "run_id": 0,
    "learning_rate": 0.01,
    "num_clients": 10,
    "num_rounds": 50,
    "local_epochs": 5,
    "batch_size": 256
  },
  "overall_metrics": {
    "f1": 0.8734,
    "accuracy": 0.8912,
    "precision": 0.8456,
    "recall": 0.9023
  },
  "per_attack_detection": {
    "Normal": {
      "detection_rate": 0.9234,
      "n_samples": 37000,
      "n_detected": 34165
    },
    "Generic": {
      "detection_rate": 0.8567,
      "n_samples": 18871,
      "n_detected": 16162
    },
    "Exploits": {
      "detection_rate": 0.7892,
      "n_samples": 33393,
      "n_detected": 26357
    },
    ...
  },
  "fairness_metrics": {
    "min_detection": 0.6234,
    "max_detection": 0.9456,
    "gap": 0.3222,
    "disparate_impact": 0.6591
  }
}
```

### Summary CSV

The `summary.csv` file contains one row per experiment with columns:
- `epsilon`, `seed`, `run_id`
- Overall metrics: `f1`, `accuracy`, `precision`, `recall`
- Fairness metrics: `min_detection`, `max_detection`, `detection_gap`, `disparate_impact`
- Per-attack detection rates: `Generic_detection`, `Exploits_detection`, etc.

## Metrics Explained

### Per-Attack Detection Rate

- **For attack types:** (# predicted as attack) / (# total samples of that type)
- **For Normal:** (# predicted as normal) / (# total normal samples)

### Fairness Metrics

Computed across attack types only (excluding Normal):

- **min_detection:** Lowest detection rate among attack types
- **max_detection:** Highest detection rate among attack types
- **gap:** max_detection - min_detection (smaller is fairer)
- **disparate_impact:** min_detection / max_detection
  - Values ≥ 0.8 are considered "fair" by the 80% rule
  - Lower values indicate larger disparity

## Validation Checks

The script automatically validates results:

1. ✓ No DP max F1 < 0.93 (matches Phase 4 baseline)
2. ✓ ε=5.0 mean F1 > 0.85 (maintains utility with moderate privacy)
3. ✓ F1 std < 0.1 across runs (results are stable)

## Interpreting Results

### Expected Patterns

1. **Overall performance:** F1 decreases as ε decreases (stronger privacy)
2. **Fairness impact:** Gap may increase with stronger privacy
3. **Attack-specific effects:** Some attack types may be more affected than others

### Key Questions to Answer

- Does the detection gap increase as privacy strengthens?
- Are certain attack types (e.g., rare types like Worms, Backdoor) disproportionately affected?
- What is the privacy-fairness tradeoff curve?
- Is disparate impact acceptable (≥ 0.8) at practical privacy levels (ε=3.0-5.0)?

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` (currently 256)
- Reduce `NUM_CLIENTS` (currently 10)

### Slow Execution
- Run experiments in parallel on multiple machines
- Reduce `NUM_ROUNDS` (currently 50) for quick testing
- Use GPU if available

### Data Not Found
- Ensure UNSW-NB15 CSV files are in the same directory as the script
- Check that files have `label` and `attack_cat` columns

### DP Errors
- If Opacus compatibility errors occur, the script uses `ModuleValidator.fix()` to auto-convert the model
- Check that `MAX_GRAD_NORM` is appropriate for your data scale

## Citation

If you use this experiment setup, please cite:

```
Federated Learning Fairness Validation Experiment
Evaluating Differential Privacy Impact on Attack-Type Detection Equity
```

## Contact

For questions or issues, please open an issue in the repository.
