# Adaptive Learning Rate for Differential Privacy

## The Problem with Strict Privacy (epsilon=0.5)

Even with MAX_GRAD_NORM=10.0, epsilon=0.5 still showed oscillation because:

**The privacy noise at epsilon=0.5 is MASSIVE**
- epsilon=0.5 is EXTREMELY strict privacy
- The noise added to gradients completely overwhelms the learning signal
- LR=0.002 (which works great for non-DP) is TOO WEAK to overcome this noise
- Result: Model learns from pure noise → random oscillation

**Think of it as:**
- epsilon=∞ (no DP): Clear vision, steady hand → LR=0.002 works fine
- epsilon=0.5 (strict DP): Blindfolded, desk shaking violently → need MUCH stronger corrections (higher LR)

## The Solution: Adaptive Learning Rate Based on Privacy Level

**Key Insight**: Stricter privacy (lower epsilon) needs higher learning rate to overcome noise.

### Implementation

**File**: `runner.py` - `BaseClient.setup_dp()`

```python
def setup_dp(self, epsilon: float, num_rounds: int):
    # Adjust LR based on epsilon
    if epsilon <= 1.0:
        adaptive_lr = 0.05  # Very strict privacy → 25x increase
    elif epsilon <= 5.0:
        adaptive_lr = 0.01  # Moderate privacy → 5x increase
    else:
        adaptive_lr = 0.005  # Relaxed privacy → 2.5x increase

    # Recreate optimizer with adaptive LR
    self.optimizer = optim.SGD(
        self.model.parameters(),
        lr=adaptive_lr,  # <-- Now adapts to privacy level!
        momentum=0.9,
        weight_decay=1e-4
    )
```

### Learning Rate Schedule by Privacy Level

| Epsilon | Privacy Level | Learning Rate | Multiplier | Rationale |
|---------|--------------|---------------|------------|-----------|
| ∞ (no DP) | None | 0.002 | 1x | Baseline, stable |
| 0.5 | VERY STRICT | **0.05** | **25x** | Massive noise needs strong signal |
| 1.0 | Strict | **0.05** | **25x** | High noise needs high LR |
| 5.0 | Moderate | **0.01** | **5x** | Moderate noise needs moderate boost |
| 10.0+ | Relaxed | **0.005** | **2.5x** | Low noise needs small boost |

## Expected Results

### epsilon=0.5 (Previously Failing):
- **Before**: Oscillating, stuck at 45% accuracy
- **After**: Should now LEARN and reach **55-60% accuracy**
- Loss should **decrease steadily** (not stuck at 0.623)
- F1 should **improve gradually** (not flip-flop)

### epsilon=1.0:
- Should reach **60-65% accuracy**
- Faster convergence than epsilon=0.5

### epsilon=5.0:
- Should reach **65-70% accuracy**
- Close to non-DP performance

### epsilon=∞ (Baseline):
- **Unchanged**: Still 70-80% accuracy
- Still uses LR=0.002

## Why This Works

**The Privacy-Utility-LR Relationship:**

1. **More privacy (lower epsilon)** = **More noise** in gradients
2. **More noise** = **Weaker signal-to-noise ratio**
3. **Weaker signal** = Need **stronger learning steps** to overcome noise
4. **Stronger learning** = **Higher learning rate**

**The Trade-off:**
- Higher LR can cause instability in non-DP training
- But with DP, the noise actually acts as regularization
- So higher LR + DP noise = stable learning!

## Testing

Run Phase 2 again:
```bash
python main.py  # With PHASE = "privacy"
```

**What to watch for:**
- Console should show: `[DP] Client X: Using adaptive LR=0.05 for epsilon=0.5`
- Loss should **decrease** over rounds (not stay at 0.623)
- Accuracy should **improve steadily** (not oscillate 45%→55%→45%)
- F1 should **gradually increase** (not flip 0→0.71→0)

**Success criteria:**
- epsilon=0.5: Reaches **55-60%** (vs 45% before)
- epsilon=1.0: Reaches **60-65%**
- epsilon=5.0: Reaches **65-70%**
- ALL show **monotonic improvement**, not oscillation

## Technical Details

**Why not just always use high LR?**
- High LR (0.05) without DP would cause oscillation (we saw this in V1!)
- The DP noise acts as implicit regularization that stabilizes high LR
- So: HighLR + DP Noise = Stable Learning
- But: High LR + No DP = Oscillation

**The Math:**
```
gradient_with_DP = true_gradient + noise(sigma)
update = LR * gradient_with_DP

For learning to work:
LR * |true_gradient| >> |noise(sigma)|

As epsilon ↓, sigma ↑ (more noise)
So LR must ↑ proportionally
```

---

**Date**: 2025-11-15
**Version**: V2.3 - Adaptive LR for DP
**Impact**: Makes strict privacy (epsilon≤1.0) actually trainable
