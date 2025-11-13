# Quick Fix for Ray OOM Errors

## Problem
You changed the config but used 'krum' aggregator which was deleted, and Ray is running out of memory.

## Solution - Apply These 3 Changes:

### 1. Fix Aggregators (REQUIRED)
```python
# ❌ WRONG (in your config):
AGGREGATORS = ['fedavg', 'krum']

# ✅ CORRECT (use this):
AGGREGATORS = ['fedavg', 'trimmed_mean']
# OR just FedAvg for fastest testing:
# AGGREGATORS = ['fedavg']
```

### 2. Limit Ray Workers (REQUIRED)
Find this line (around line 895 in nwdaf_security_analytics_.py):
```python
ray.init(num_cpus=os.cpu_count(), log_to_driver=False, ignore_reinit_error=True)
```

Change to:
```python
# Only run 2 experiments at a time to avoid OOM
ray.init(num_cpus=2, log_to_driver=False, ignore_reinit_error=True)
```

### 3. Verify Your Test Config
Your configuration should look like this:
```python
NUM_CLIENTS = 3              # ✅ Good
NUM_ROUNDS = 10              # ✅ Good
NUM_RUNS = 1                 # ✅ Good
BATCH_SIZE = 128             # ✅ Good

EPSILON_VALUES = [5.0, float('inf')]      # ✅ Good (2 values)
MALICIOUS_FRACTIONS = [0.0, 0.2]          # ✅ Good (2 values)
ATTACK_TYPES = ['none', 'label_flip']     # ✅ Good (2 types)
AGGREGATORS = ['fedavg', 'trimmed_mean']  # ✅ FIXED (was using 'krum')
```

## Expected Results After Fix:
- **Total experiments**: 8 (2 eps × 2 alpha × 2 agg × 1 run)
- **Runtime**: 10-30 minutes
- **Memory usage**: Should stay under 8GB
- **No more OOM errors**

## Alternative: Run Just FedAvg (Fastest)
If you want the absolute fastest test:
```python
AGGREGATORS = ['fedavg']  # Just baseline, no robust aggregation
```
This gives you 4 experiments total, completes in ~5-15 minutes.

## Still Getting Errors?
If you still see OOM after these fixes:
1. Reduce to `ray.init(num_cpus=1, ...)` - runs one at a time
2. Reduce NUM_CLIENTS to 2
3. Reduce BATCH_SIZE to 64
4. Check: `free -h` to see available memory
