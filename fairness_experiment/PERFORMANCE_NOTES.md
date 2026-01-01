# Performance Notes & Troubleshooting

## Issue: Training Appears Stuck at Round 1

### Root Cause
The training is **not stuck** - it's just **very slow** due to:

1. **Feature Dimension Increase**: One-hot encoding of categorical columns (`proto`, `service`, `state`) increased features from 45 → 194 (4.3x)
2. **Opacus DP Overhead**: Differential privacy requires per-sample gradient computation, which is 10-20x slower than normal training
3. **Larger Model**: 194 input features → larger weight matrices → more computation per batch

### Performance Impact

| Configuration | Expected Time per Round |
|--------------|------------------------|
| No DP (ε=∞) | 30-60 seconds |
| DP (ε=5.0) | 5-10 minutes |
| DP (ε=3.0) | 8-15 minutes |
| DP (ε=1.0) | 10-20 minutes |

**Total experiment runtime estimates:**
- Quick test (5 rounds, ε=5.0): 25-50 minutes
- Single full experiment (50 rounds, ε=5.0): 4-8 hours
- All 12 experiments: 48-96 hours

## Diagnostic Tools

### 1. Test Without DP (Fastest)
```bash
cd fairness_experiment
python diagnostic_test.py
```
Tests basic training without DP overhead. Should complete in 2-3 minutes.

### 2. Optimized Quick Test (Recommended)
```bash
python quick_test_optimized.py
```
Reduced parameters for faster validation:
- 3 clients (vs 10)
- 3 rounds (vs 50)
- 2 epochs (vs 5)
- Batch size 128 (vs 256)
- Expected runtime: 3-5 minutes

### 3. Original Quick Test
```bash
python quick_test.py
```
More thorough but slower:
- 5 clients
- 5 rounds
- 3 epochs
- Expected runtime: 25-50 minutes

## Progress Monitoring

The updated code now includes progress logging. You should see output like:
```
  Epoch 1/3, Batch 10, Avg Loss: 0.4523
  Epoch 1/3, Batch 20, Avg Loss: 0.4312
  Epoch 1/3, Batch 30, Avg Loss: 0.4156
```

If you see these logs appearing every few seconds, training is progressing normally.

## Optimization Recommendations

### For Testing/Development
1. Use `quick_test_optimized.py` for quick validation
2. Test with ε=∞ (no DP) first using `diagnostic_test.py`
3. Reduce `NUM_CLIENTS` and `NUM_ROUNDS` for faster iteration

### For Production Runs
1. **Use GPU**: Essential for reasonable performance
2. **Reduce batch size**: Smaller batches = faster per-batch computation
   - Try BATCH_SIZE = 128 or even 64
   - Trade-off: More batches per epoch, but each batch is faster
3. **Parallelize experiments**: Run different ε values on different machines
4. **Be patient**: DP training is inherently slow

### Code Modifications for Speed

#### Reduce Batch Size
Edit `run_fairness_validation.py`:
```python
BATCH_SIZE = 128  # or 64 for even faster batches
```

#### Reduce Clients/Rounds for Testing
```python
NUM_CLIENTS = 5   # instead of 10
NUM_ROUNDS = 25   # instead of 50
```

#### Skip DP for Initial Testing
```python
EPSILON_VALUES = [float('inf')]  # No DP
```

## Understanding the Slowdown

### Why is DP Training So Slow?

**Normal Training:**
- Compute gradients for entire batch at once
- Single backward pass per batch

**DP Training (Opacus):**
- Compute gradients **per sample** (per-sample gradient clipping)
- 256 samples = 256 individual backward passes
- Apply Gaussian noise to clipped gradients
- 10-20x slower

**With Larger Feature Dimension:**
- 194 features → larger weight matrices
- More computation per sample
- More memory usage
- Even slower

### Why Not Stuck?

The code is working correctly. The progress logs show:
```
[ROUND 1]
(pid=16852) ...  # Client processes starting
(pid=16853) ...  # Multiple clients training
```

These indicate Ray is successfully distributing work to clients. The long pause is normal DP training overhead.

## Expected Behavior

### What You'll See (Normal):
1. Quick startup and data loading (< 30 seconds)
2. Ray initialization (< 30 seconds)
3. **Long pause at [ROUND 1]** (5-10 minutes for DP)
4. Progress logs appearing (every 10 batches)
5. Round completion
6. Repeat for each round

### What Indicates a Problem:
1. No progress logs for > 15 minutes
2. Out of memory errors
3. GPU not detected when expected
4. Python process crashes

## Verification Steps

1. **Check GPU Usage**: `nvidia-smi` should show GPU memory in use
2. **Check CPU Usage**: Training processes should show 100% CPU
3. **Watch for Progress Logs**: Should appear every 10-30 seconds
4. **Check Memory**: Ensure sufficient RAM (16GB+ recommended)

## Recommendations by Use Case

### Just Want to Verify Setup Works
→ Use `diagnostic_test.py` (2-3 minutes, no DP)

### Want Quick DP Validation
→ Use `quick_test_optimized.py` (3-5 minutes)

### Want Results Close to Full Experiment
→ Use `quick_test.py` but be patient (25-50 minutes)

### Running Full Experiment
→ Use `run_fairness_validation.py` and plan for 48-96 hours total
→ Consider parallel execution on multiple machines

## Still Having Issues?

If training appears genuinely stuck:

1. Check Ray logs for errors
2. Verify GPU is available: `torch.cuda.is_available()`
3. Try `diagnostic_test.py` without DP
4. Check system resources (memory, disk space)
5. Look for out-of-memory errors in logs

## Future Optimizations

Potential improvements for future versions:

1. **Proper noise calibration**: Calculate optimal noise_multiplier based on target ε
2. **Gradient accumulation**: Reduce memory usage
3. **Mixed precision training**: Use FP16 for faster computation
4. **Vectorized DP**: Use newer Opacus features for faster per-sample gradients
5. **Async client updates**: Don't wait for all clients synchronously
