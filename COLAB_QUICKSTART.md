# Google Colab Quick Start Guide

## Step-by-Step Instructions for Running FL-NIDS Experiments

### 1. Setup in Google Colab

```python
# Mount Google Drive (if data is there)
from google.colab import drive
drive.mount('/content/drive')

# Clone or upload your code
# Option A: Upload files directly to Colab
# Option B: Clone from GitHub
# !git clone https://github.com/your-repo/FL_research.git
# %cd FL_research

# Install dependencies
!pip install flwr opacus scikit-learn pandas numpy scipy matplotlib seaborn
```

### 2. Verify Data Path

```python
# Check if data exists
import os
data_path = '/content/drive/MyDrive/IDSDatasets/UNSW 15'
print("Data path exists:", os.path.exists(data_path))
print("Files:", os.listdir(data_path) if os.path.exists(data_path) else "NOT FOUND")
```

### 3. Configure Experiment

```python
# Edit config.py
# Change PHASE variable to one of: baseline, privacy, aggregators, attacks, full

# For quick testing (recommended first):
# - Set NUM_RUNS = 1 (instead of 5)
# - Set NUM_ROUNDS = 20 (instead of 100)
# - Set use_sample = True, sample_fraction = 0.1 (use 10% of data)

# Example edits in config.py:
```

```python
# Quick test configuration (edit in config.py)
PHASE = "baseline"
NUM_RUNS = 1
NUM_ROUNDS = 20

# In get_phase_config(), add:
'use_sample': True,
'sample_fraction': 0.1,
```

### 4. Run Experiments

```python
# Run main script
!python main.py
```

### 5. Check Results

```python
# List result files
!ls -lh results/phase1_baseline/
!ls -lh results/phase1_baseline/analysis/

# View summary
import json
with open('results/phase1_baseline/analysis/statistical_analysis_final_f1.json', 'r') as f:
    analysis = json.load(f)

print("Best configurations:")
for i, config in enumerate(analysis['best_configurations'][:5]):
    print(f"{i+1}. {config['config']}: {config['mean']:.4f} ± {config['std']:.4f}")
```

### 6. View Individual Experiment

```python
# Load a specific experiment
with open('results/phase1_baseline/eps_inf_fedavg_clean_run0.json', 'r') as f:
    exp = json.load(f)

print("Final F1:", exp['summary']['final_f1'])
print("Convergence round:", exp['summary']['convergence_round'])
print("Total time:", exp['summary']['total_time'])

# Plot F1 over rounds
import matplotlib.pyplot as plt
rounds = [m['round'] for m in exp['round_metrics']]
f1_scores = [m.get('f1', 0) for m in exp['round_metrics']]

plt.plot(rounds, f1_scores)
plt.xlabel('Round')
plt.ylabel('F1 Score')
plt.title('F1 Score over Training Rounds')
plt.grid(True)
plt.show()
```

### 7. Run Full Experiment (After Testing)

Once you've verified everything works:

```python
# Edit config.py:
# - Set NUM_RUNS = 5
# - Set NUM_ROUNDS = 100
# - Set use_sample = False

# Then run:
!python main.py
```

### 8. Handle Disconnections

If Colab disconnects, just reconnect and run again:

```python
# Reconnect to Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate back to directory
%cd /content/FL_research  # or wherever your code is

# Run again - it will resume from checkpoint
!python main.py
```

Output will show:
```
Total experiments: 35
Completed: 12
Remaining: 23
↻ Resuming from checkpoint (12 experiments done)
```

### 9. Download Results

```python
# Compress results for download
!tar -czf results.tar.gz results/

# Download via Colab
from google.colab import files
files.download('results.tar.gz')
```

Or save to Drive:

```python
# Copy results to Drive
!cp -r results /content/drive/MyDrive/FL_NIDS_Results/
```

## Recommended Testing Sequence

### Test 1: Quick Validation (5 minutes)
```python
PHASE = "baseline"
NUM_RUNS = 1
NUM_ROUNDS = 10
use_sample = True
sample_fraction = 0.1
```

### Test 2: Single Config Full Run (20 minutes)
```python
PHASE = "baseline"
NUM_RUNS = 3
NUM_ROUNDS = 50
use_sample = False
# Keep only 1 config in baseline
```

### Test 3: Full Baseline Phase (2 hours)
```python
PHASE = "baseline"
NUM_RUNS = 5
NUM_ROUNDS = 100
use_sample = False
```

### Test 4: Full Privacy Analysis (4 hours)
```python
PHASE = "privacy"
NUM_RUNS = 5
NUM_ROUNDS = 100
use_sample = False
```

## Troubleshooting in Colab

### Memory Issues

```python
# Check memory
!free -h

# If running out of memory:
# 1. Use GPU runtime (Runtime > Change runtime type > GPU)
# 2. Reduce clients:
NUM_CLIENTS = 5  # instead of 10

# 3. Reduce batch size:
BATCH_SIZE = 64  # instead of 128

# 4. Use sampling:
use_sample = True
sample_fraction = 0.3
```

### Slow Execution

```python
# Enable GPU acceleration
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# If GPU not available, request GPU runtime:
# Runtime > Change runtime type > Hardware accelerator > GPU
```

### Monitor Progress

```python
# In another cell, monitor progress
import time
import json
from pathlib import Path

while True:
    completed = len(list(Path('results/phase1_baseline').glob('*.json')))
    print(f"Completed experiments: {completed}", end='\r')
    time.sleep(10)
```

### Save Intermediate Results

```python
# Periodically save to Drive
import shutil
import time

# Run this in parallel with main.py
while True:
    try:
        shutil.copytree('results', '/content/drive/MyDrive/FL_NIDS_Results_Backup',
                       dirs_exist_ok=True)
        print(f"Backup saved at {time.strftime('%H:%M:%S')}")
    except:
        pass
    time.sleep(300)  # Every 5 minutes
```

## Expected Runtimes (Colab GPU)

| Phase | Configs | Total Experiments | Estimated Time |
|-------|---------|-------------------|----------------|
| Baseline | 3 | 15 (3×5) | 1-2 hours |
| Privacy | 7 | 35 (7×5) | 3-4 hours |
| Aggregators | 24 | 120 (24×5) | 10-12 hours |
| Attacks | 27 | 135 (27×5) | 12-15 hours |
| Full | 96 | 480 (96×5) | 40-50 hours |

**Tip**: For phases >12 hours, split into smaller chunks or use Colab Pro.

## Tips for Long Experiments

1. **Use Colab Pro**: Longer runtime limits, better GPUs
2. **Run overnight**: Start before sleep, check in morning
3. **Monitor with phone**: Install Colab mobile app
4. **Enable notifications**: Get alerts when cells complete
5. **Keep browser open**: Prevents early disconnections
6. **Disable sleep**: Prevent laptop from sleeping

## Verifying Results Quality

```python
# Check if experiments completed successfully
import json
from pathlib import Path

results_dir = Path('results/phase1_baseline')
files = list(results_dir.glob('*.json'))

print(f"Total result files: {len(files)}")

# Check each file
for f in files:
    with open(f) as fp:
        data = json.load(fp)
        summary = data['summary']
        print(f"{f.name}:")
        print(f"  Rounds: {summary['final_round']}")
        print(f"  Final F1: {summary['final_f1']:.4f}")
        print(f"  Converged: {summary.get('converged', False)}")
        print()
```

## Common Errors

### "No module named 'flwr'"
```python
!pip install flwr opacus
```

### "CUDA out of memory"
```python
# Edit config.py:
NUM_CLIENTS = 5
BATCH_SIZE = 64
```

### "Could not find UNSW-NB15 dataset"
```python
# Check and fix DATA_PATH in config.py
# Make sure Drive is mounted
```

### "Connection timeout"
```python
# Colab disconnected - just reconnect and run again
# Experiments will resume automatically
```

Good luck! 🚀
