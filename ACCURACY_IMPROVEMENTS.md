# Accuracy Improvements - Summary (Version 2)

## Problem Statement

The model was **not learning** during federated training with severe oscillation:
- Final accuracy: **44.9%** (essentially random)
- Final F1 score: **0.0** (predicting only majority class)
- Loss stuck at **~0.623** across all 50 rounds
- **Model oscillating between exactly two states** - critical symptom
- Oscillates between F1=0.0 (acc=44.9%) and F1=0.71 (acc=55%)
- Best F1 was 0.71 at round 2, then degraded

## Root Cause Analysis

### Primary Issue: INSTABILITY IN FEDERATED TRAINING
The oscillation pattern revealed fundamental instability:
1. **Adam optimizer + High LR causing oscillation**
   - Initial fix of LR=0.01 was TOO HIGH for federated setting
   - Each client overshoots during local training
   - Aggregated weights bounce between states instead of converging

2. **Complex Model Architecture**
   - 3-layer network [128, 64, 32] too complex for stable convergence
   - GroupNorm may not work well with federated aggregation
   - Too many parameters to stabilize with limited local data

3. **Aggressive Hyperparameters**
   - Too many local epochs (10) with high LR causes massive overshooting
   - Small batch size (128) creates high variance in gradients
   - Tight gradient clipping (1.0) may interfere with learning

## Implemented Fixes (V2 - STABILITY-FOCUSED)

### 1. Switch to SGD with Momentum ⭐ CRITICAL
**File**: `runner.py` - `BaseClient.__init__()`
```python
self.optimizer = optim.SGD(
    model.parameters(),
    lr=config.LEARNING_RATE,  # 0.002
    momentum=0.9,
    weight_decay=1e-4
)
```
**Impact**: SGD more stable than Adam for federated learning, consistent momentum helps convergence

### 2. Conservative Learning Rate
**File**: `config.py`
```python
LEARNING_RATE = 0.002  # Conservative, stable learning
```
**Impact**: Prevents oscillation while still enabling progress

### 3. Simplified Model Architecture
**File**: `config.py` and `runner.py`
```python
HIDDEN_DIMS = [64, 32]  # Reduced from [128, 64, 32]
```
**Impact**: Simpler model is easier to train and stabilize in federated setting

### 4. BatchNorm Instead of GroupNorm
**File**: `runner.py` - `IntrusionDetectionMLP`
```python
nn.BatchNorm1d(hidden_dim)  # Instead of GroupNorm
```
**Impact**: BatchNorm better suited for federated learning, standard approach

### 5. Larger Batch Size
**File**: `config.py`
```python
BATCH_SIZE = 256  # Increased from 128
```
**Impact**: Reduces gradient variance, more stable updates

### 6. Moderate Local Epochs
**File**: `config.py`
```python
LOCAL_EPOCHS = 5  # Balanced training per round
```
**Impact**: Enough training without overshooting

### 7. Minimal Dropout
**File**: `config.py`
```python
DROPOUT_RATE = 0.1  # Reduced from 0.3
```
**Impact**: Less regularization during learning phase

### 8. Gentler Gradient Clipping
**File**: `runner.py`
```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
```
**Impact**: Prevents explosions without overly constraining gradients

### 9. Removed Learning Rate Scheduler
**File**: `runner.py`
```python
# Scheduler disabled - can interfere with FL
```
**Impact**: Consistent learning rate across all clients and rounds

## Expected Improvements

### Performance Metrics:
- **Final Accuracy**: Should reach **70-80%** (vs current 44.9%)
- **Final F1 Score**: Should reach **0.60-0.75** (vs current 0.0)
- **Loss Reduction**: Should decrease steadily from ~0.62 to **0.35-0.45**
- **Convergence**: Should converge within **25-35 rounds** (vs not converging)

### Training Behavior:
- **NO MORE OSCILLATION** - steady monotonic improvement
- Consistent F1 score increases (not bouncing between 0 and 0.71)
- Balanced predictions across both classes
- Stable loss decrease without sudden jumps
- All clients contribute meaningfully to global model

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

## Changelog

### Version 2 (Current) - STABILITY FIX
After V1 still showed oscillation, diagnosed as:
- Adam optimizer instability in FL setting
- Model too complex for stable convergence
- Learning rate still too high

**V2 Changes**:
- SGD instead of Adam
- Simpler architecture [64, 32]
- LR = 0.002 (conservative)
- Larger batch size (256)
- Removed scheduler
- BatchNorm instead of GroupNorm

### Version 1 - INITIAL FIX (Unsuccessful)
- Increased LR to 0.01 (caused oscillation)
- Added scheduler (may have interfered)
- Increased local epochs to 10 (too many)
- Complex initialization (unnecessary)

## Files Modified

1. **config.py**: Hyperparameters (LR=0.002, epochs=5, batch=256, simpler model)
2. **runner.py**: SGD optimizer, simplified architecture, BatchNorm
3. **debug_training.py**: Updated to match configuration
4. **ACCURACY_IMPROVEMENTS.md**: This documentation

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
