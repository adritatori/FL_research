# Fairness Experiment Files Overview

## Quick Answer

**You only need the files in `fairness_experiment/` directory - they are completely standalone!**

No previous files are required. The experiment is self-contained.

---

## File Structure

### New Experiment (Standalone)
```
fairness_experiment/
├── run_fairness_validation.py   ← Main experiment script (STANDALONE)
├── README.md                     ← Documentation
├── requirements.txt              ← Dependencies
└── quick_test.py                 ← Setup verification
```

### Existing File (Different Purpose)
```
fairness_validation.py            ← Previous Phase 4 experiment (NOT NEEDED)
```

---

## Key Differences

| Aspect | New Experiment | Old Experiment |
|--------|---------------|----------------|
| **Location** | `fairness_experiment/` | Root directory |
| **Purpose** | Isolate DP impact on fairness | Byzantine attacks + DP |
| **Attack scenario** | Clean (no Byzantine) | 40% malicious clients |
| **Configurations** | 4 privacy × 3 runs = 12 | 4 experiments |
| **Dependencies** | None (standalone) | N/A |
| **Data path** | `/content/drive/MyDrive/IDSDatasets/UNSW 15` | Same |

---

## What You Need

### 1. Files (already created ✓)
- `fairness_experiment/run_fairness_validation.py`
- `fairness_experiment/README.md`
- `fairness_experiment/requirements.txt`
- `fairness_experiment/quick_test.py`

### 2. Data (you should have this)
Location: `/content/drive/MyDrive/IDSDatasets/UNSW 15/`
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

### 3. Dependencies
```bash
pip install torch numpy pandas scikit-learn flwr opacus
```

---

## How to Run

### Quick Test (5-10 min)
```bash
cd fairness_experiment
python quick_test.py
```

### Full Experiment (6-8 hours)
```bash
cd fairness_experiment
python run_fairness_validation.py
```

---

## Data Path Configuration

✅ **Already configured** for your Google Drive path:
```python
DATA_DIR = Path("/content/drive/MyDrive/IDSDatasets/UNSW 15")
```

If you need to change it, edit line 95 in `run_fairness_validation.py`.

---

## Why This Experiment is Different

**Research Question:** Does DP disproportionately harm certain attack types?

**Approach:**
- Run FL with different privacy levels (ε = ∞, 5.0, 3.0, 1.0)
- **No Byzantine attacks** (clean federated learning)
- Measure detection rate for each of 9 attack types
- Compare fairness metrics: gap, disparate impact

**Output:**
- Per-attack-type detection rates
- Fairness analysis
- 12 JSON result files
- Model weights
- Summary CSV

This isolates the effect of differential privacy on fairness, without confounding from Byzantine attacks.

---

## Summary

✅ **Standalone** - No dependencies on other files
✅ **Data path** - Already configured for your Google Drive
✅ **Ready to run** - Just install dependencies and execute
✅ **Clean design** - Focuses only on DP impact on fairness

The old `fairness_validation.py` in the root directory serves a different purpose (Byzantine attacks + DP). The new experiment is completely independent.
