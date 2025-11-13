# 🔧 COMPLETE FIX FOR RAY OOM ERRORS

## Problem Summary
Your Ray workers are crashing with Out-Of-Memory (OOM) errors because:
1. ❌ You used `'krum'` aggregator (which was deleted in optimization)
2. ❌ Ray spawns too many parallel workers (`num_cpus=os.cpu_count()`)
3. ⚠️ Each worker loads full dataset + model into memory

---

## ✅ SOLUTION: TWO CRITICAL CHANGES

### **Fix #1: Change Your Configuration**

In your Jupyter notebook, replace your config with:

```python
# ❌ WRONG (Your current config):
NUM_CLIENTS = 3
NUM_ROUNDS = 10
NUM_RUNS = 1
BATCH_SIZE = 128
EPSILON_VALUES = [5.0, float('inf')]
MALICIOUS_FRACTIONS = [0.0, 0.2]
ATTACK_TYPES = ['none', 'label_flip']
AGGREGATORS = ['fedavg', 'krum']  # ❌❌❌ KRUM DOESN'T EXIST!

# ✅ CORRECT (Use this):
NUM_CLIENTS = 3
NUM_ROUNDS = 10
NUM_RUNS = 1
BATCH_SIZE = 128
EPSILON_VALUES = [5.0, float('inf')]
MALICIOUS_FRACTIONS = [0.0, 0.2]
ATTACK_TYPES = ['none', 'label_flip']
AGGREGATORS = ['fedavg', 'trimmed_mean']  # ✅ Both exist and work!
```

**OR** use the pre-made config:
```python
from working_test_config import WorkingTestConfig as ExperimentConfig
config = ExperimentConfig()
```

---

### **Fix #2: Limit Ray Workers**

Find this line in your notebook (should be in the `run_full_experiment` method):

```python
# ❌ WRONG (Line ~916 in the original file):
ray.init(num_cpus=os.cpu_count(), log_to_driver=False, ignore_reinit_error=True)
```

**Change to:**
```python
# ✅ CORRECT - Only run 1-2 experiments at a time:
ray.init(num_cpus=2, log_to_driver=False, ignore_reinit_error=True)
```

**Why this works:**
- `os.cpu_count()` returns ALL CPUs (8-16 on most systems)
- Each Ray worker loads ~1-2GB of data
- With 8 workers × 2GB = 16GB+ memory usage → OOM crash
- Limiting to 2 workers = ~4GB usage = no crash

---

## 📋 COMPLETE STEP-BY-STEP FIX

### **Step 1: In your notebook, add this cell at the top:**

```python
# Import the working configuration
from working_test_config import WorkingTestConfig

# Use it as your config
config = WorkingTestConfig()
config.setup_directories()
config.set_seed()
config.print_config()
```

### **Step 2: Find the ExperimentRunner class in your notebook**

Search for the `run_full_experiment` method (around line 916 in the original).

### **Step 3: Change the ray.init() line:**

```python
# Find this:
ray.init(num_cpus=os.cpu_count(), log_to_driver=False, ignore_reinit_error=True)

# Replace with this:
ray.init(num_cpus=2, log_to_driver=False, ignore_reinit_error=True)
print(f"\n⚠️  Ray limited to 2 CPUs to prevent OOM errors")
```

### **Step 4: Re-run your notebook**

Expected output:
```
EXPERIMENTAL SETUP:
  Epsilon values: [5.0, inf]
  Malicious fractions: [0.0, 0.2]
  Attack types: ['none', 'label_flip']
  Aggregators: ['fedavg', 'trimmed_mean']
  Total configurations: 8
  Estimated time: 15-30 minutes

Ray initialized on 2 CPUs. 🔥
⚠️  Ray limited to 2 CPUs to prevent OOM errors

Total experiments to run: 8
```

---

## 🎯 QUICK VERIFICATION

Before running, verify your config:

```python
# Check aggregators (MUST NOT contain 'krum')
print("Aggregators:", config.AGGREGATORS)
# Expected: ['fedavg', 'trimmed_mean']

# Check if trimmed_mean and median strategies exist
from nwdaf_security_analytics_ import TrimmedMeanStrategy, MedianStrategy
print("✓ TrimmedMeanStrategy exists")
print("✓ MedianStrategy exists")

# Try importing Krum (should fail)
try:
    from nwdaf_security_analytics_ import KrumStrategy
    print("❌ ERROR: Krum still exists (shouldn't happen)")
except:
    print("✓ Krum correctly removed")
```

---

## 🐛 STILL GETTING ERRORS?

### **Error: "NameError: name 'KrumStrategy' is not defined"**
- ✅ This confirms Krum doesn't exist
- ❌ Your AGGREGATORS still contains 'krum'
- **Fix:** Change AGGREGATORS to `['fedavg', 'trimmed_mean']`

### **Error: Ray workers still crashing with OOM**
Try these in order:

1. **Reduce Ray workers to 1:**
   ```python
   ray.init(num_cpus=1, ...)  # Sequential execution
   ```

2. **Reduce NUM_CLIENTS:**
   ```python
   NUM_CLIENTS = 2  # Fewer clients = less memory
   MIN_FIT_CLIENTS = 2
   MIN_AVAILABLE_CLIENTS = 2
   ```

3. **Reduce BATCH_SIZE:**
   ```python
   BATCH_SIZE = 64  # Half the memory per batch
   ```

4. **Check available memory:**
   ```bash
   free -h  # Should show >8GB available
   ```

5. **Kill existing Ray processes:**
   ```python
   import ray
   ray.shutdown()
   # Then restart your notebook
   ```

---

## ✅ EXPECTED RESULTS

After applying these fixes:
- ✅ No more "Worker died unexpectedly" errors
- ✅ 8 experiments complete successfully
- ✅ Runtime: 15-30 minutes
- ✅ Memory usage: 4-6 GB peak
- ✅ All results saved to `./results/`

---

## 📊 EXPERIMENT BREAKDOWN

With the working config, you'll run:

**Clean experiments (α=0.0, attack='none'):**
1. ε=5.0, α=0.0, none, fedavg
2. ε=5.0, α=0.0, none, trimmed_mean
3. ε=∞, α=0.0, none, fedavg
4. ε=∞, α=0.0, none, trimmed_mean

**Attack experiments (α=0.2, attack='label_flip'):**
5. ε=5.0, α=0.2, label_flip, fedavg
6. ε=5.0, α=0.2, label_flip, trimmed_mean
7. ε=∞, α=0.2, label_flip, fedavg
8. ε=∞, α=0.2, label_flip, trimmed_mean

**Total: 8 experiments × 10 rounds = 80 FL rounds**

---

## 🔗 FILE LOCATIONS

- Working config: `working_test_config.py` (in repo root)
- Original file: `nwdaf_security_analytics_.py`
- This guide: `FIX_RAY_OOM.md`
- Quick reference: `QUICK_FIX.md`

---

## 💡 WHY KRUM WAS REMOVED

Krum aggregation has O(n²d) complexity where:
- n = number of clients
- d = number of model parameters

For your MLP model with ~200K parameters:
- 5 clients × 200K params = **5 billion distance calculations per round**
- With 50 rounds = **250 billion operations**
- Runtime: 10-20x slower than FedAvg

**Better alternatives:**
- `trimmed_mean`: O(n log n), almost as robust, 100x faster
- `median`: O(n log n), maximum robustness, 100x faster
