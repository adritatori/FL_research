# Accuracy Improvements - Summary

## Problem Statement

The model was **not learning** during federated training:
- Final accuracy: **44.9%** (essentially random)
- Final F1 score: **0.0** (predicting only majority class)
- Loss stuck at **~0.623** across all 50 rounds
- Model oscillating between two states without improvement
- Best F1 was 0.71 at round 2, then degraded

## Root Cause Analysis

### Primary Issues:
1. **Learning Rate Too Low (0.001)**
   - With 10 clients and federated averaging, effective learning rate was even smaller
   - Model couldn't escape initial state or local minima
   - Gradients too small to make meaningful parameter updates

2. **Insufficient Training Per Round**
   - Only 5 local epochs per client
   - Combined with low learning rate, clients made minimal progress

3. **Poor Weight Initialization**
   - Default PyTorch initialization not optimal for deep networks
   - All clients started with identical weights

### Secondary Issues:
1. **High Dropout (0.3)** - Too aggressive regularization during learning phase
2. **No Learning Rate Scheduling** - Fixed LR prevented adaptive learning
3. **No Gradient Monitoring** - Couldn't detect vanishing/exploding gradients

## Implemented Fixes

### 1. Learning Rate Increase ⭐ CRITICAL
**File**: `config.py`
```python
LEARNING_RATE = 0.01  # Increased from 0.001 (10x increase)
```
**Impact**: Enables model to make meaningful updates and escape local minima

### 2. Increased Local Training
**File**: `config.py`
```python
LOCAL_EPOCHS = 10  # Increased from 5
```
**Impact**: Each client trains longer per round, making better use of local data

### 3. Reduced Dropout
**File**: `config.py`
```python
DROPOUT_RATE = 0.2  # Reduced from 0.3
```
**Impact**: Less aggressive regularization allows better learning

### 4. Learning Rate Scheduling
**File**: `runner.py` - `BaseClient.__init__()`
```python
self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    self.optimizer, mode='min', factor=0.5, patience=3, verbose=False
)
```
**Impact**: Adapts learning rate based on loss, enables fine-tuning

### 5. Weight Decay (L2 Regularization)
**File**: `runner.py` - `BaseClient.__init__()`
```python
self.optimizer = optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE,
    betas=(0.9, 0.999),
    weight_decay=1e-5  # Added L2 regularization
)
```
**Impact**: Prevents overfitting and improves generalization

### 6. Gradient Clipping
**File**: `runner.py` - `BaseClient.fit()`
```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```
**Impact**: Prevents exploding gradients, stabilizes training

### 7. Improved Weight Initialization
**File**: `runner.py` - `IntrusionDetectionMLP.__init__()`
```python
# Kaiming (He) initialization for ReLU layers
nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
nn.init.constant_(linear.bias, 0.01)

# Xavier initialization for output layer
nn.init.xavier_normal_(output_layer.weight)
nn.init.constant_(output_layer.bias, 0.0)
```
**Impact**: Better initial weights lead to faster convergence

### 8. Enhanced Monitoring
**File**: `runner.py` - `BaseClient.fit()`
- NaN loss detection
- Learning rate tracking in metrics
- Per-epoch loss tracking
- Scheduler updates based on validation loss

## Expected Improvements

### Performance Metrics:
- **Final Accuracy**: Should reach **70-85%** (vs current 44.9%)
- **Final F1 Score**: Should reach **0.65-0.80** (vs current 0.0)
- **Loss Reduction**: Should decrease from ~0.62 to **0.3-0.4**
- **Convergence**: Should converge within **20-30 rounds** (vs not converging)

### Training Behavior:
- Steady loss decrease instead of oscillation
- Consistent improvement in F1 score
- Balanced predictions (not just majority class)
- Gradual learning rate reduction as model converges

## Testing Recommendations

1. **Run Baseline Test** (Quick validation):
   ```bash
   python main.py
   ```
   Monitor first 10 rounds - should see:
   - Loss decreasing from ~0.62
   - F1 score improving from round 2-3
   - Accuracy increasing steadily

2. **Check Debug Script**:
   ```bash
   python debug_training.py
   ```
   Verify centralized training works well

3. **Compare Results**:
   - Previous best F1: **0.71** (round 2)
   - New expected best F1: **>0.75** (within 15 rounds)

## Files Modified

1. **config.py**: Core hyperparameters (LR, epochs, dropout)
2. **runner.py**: Training loop, initialization, scheduling
3. **debug_training.py**: Updated to match new configuration

## Rollback Instructions

If these changes cause issues, revert with:
```bash
git checkout HEAD~1 config.py runner.py debug_training.py
```

Original values:
- `LEARNING_RATE = 0.001`
- `LOCAL_EPOCHS = 5`
- `DROPOUT_RATE = 0.3`
- No scheduler, no weight decay, no gradient clipping

## Next Steps

1. Run experiments and monitor metrics
2. If accuracy is still low (<60%), try:
   - Increase learning rate to 0.02
   - Increase local epochs to 15
   - Adjust batch size to 64
3. If overfitting occurs, try:
   - Increase dropout to 0.25
   - Increase weight decay to 5e-5
4. For fine-tuning, adjust scheduler patience to 2-3 rounds

---
**Date**: 2025-11-15
**Author**: Claude
**Issue**: Poor model accuracy (44.9% → target 70-85%)
**Status**: Ready for testing
