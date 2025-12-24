# Section 4: Experimental Setup - Complete Parameters

## Overview
You have **TWO separate experimental configurations**:
1. **Threat Analysis (Phase 4)** - Comprehensive attack scenarios with DP and robust aggregators
2. **Fairness Validation** - Per-attack-type detection rate analysis

---

## 4.A Dataset and Preprocessing

### Dataset Selection: UNSW-NB15

**Why UNSW-NB15?**
- Modern attack types (2015) vs outdated KDD/NSL-KDD (1999)
- Realistic network traffic with 9 attack categories
- Binary classification: Normal (0) vs Attack (1)
- Better class balance than CIC-IDS variants
- 175,341 total samples (after combining train+test sets)

**Attack Categories (10 classes):**
- Normal: Benign traffic (class 0)
- Generic: 40,000 samples (18.0%)
- Exploits: 33,393 samples (15.0%)
- Fuzzers: 18,184 samples (8.2%)
- DoS: 12,264 samples (5.5%)
- Reconnaissance: 10,491 samples (4.7%)
- Analysis: 2,000 samples (0.9%) - RARE
- Backdoor: 1,746 samples (0.8%) - RARE
- Shellcode: 1,133 samples (0.5%) - RARE
- Worms: 130 samples (0.06%) - VERY RARE

### Preprocessing Pipeline

1. **Feature Engineering:**
   - Input features: 196 (after dropping 'id', 'attack_cat', 'label')
   - Categorical encoding: LabelEncoder for ['proto', 'service', 'state']
   - Numerical conversion with missing value handling (0-fill)
   - Infinite value clipping: posinf → 1e10, neginf → -1e10

2. **Train-Test Split:**
   - Test size: 20% (35,068 samples)
   - Training size: 80% (140,273 samples)
   - Stratified split by binary label
   - Random seed: 42

3. **Normalization:**
   - StandardScaler (zero mean, unit variance)
   - Fitted on training set, applied to test set

4. **Data Partitioning (Federated Setup):**
   - **IID split** across 10 clients
   - Random uniform sampling (not stratified at client level)
   - Each client receives ~14,027 samples
   - No client-specific attack distributions

---

## 4.B Federated Learning Configuration

### FL Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Number of clients (N)** | 10 | Fixed across all experiments |
| **Client fraction** | 1.0 | All clients participate per round |
| **Min fit clients** | 8 | Minimum for aggregation |
| **FL rounds (T)** | 50 | Maximum (early stopping enabled) |
| **Local epochs (E)** | 5 | Per-client training epochs |
| **Batch size (B)** | 256 | For local SGD |
| **Learning rate (η)** | 0.002 (base) | Adaptive for DP (see 4.C) |
| **Optimizer** | SGD + Momentum | momentum=0.9, weight_decay=1e-4 |
| **Early stopping** | Patience=10 | Convergence threshold: F1 ≥ 0.90 |

### Model Architecture

**Binary MLP Classifier:**
```
Input layer: 196 features
  ↓
Hidden layer 1: 64 neurons
  → GroupNorm(num_groups=1) [DP-compatible, equivalent to LayerNorm]
  → ReLU
  → Dropout(p=0.1)
  ↓
Hidden layer 2: 32 neurons
  → GroupNorm(num_groups=1)
  → ReLU
  → Dropout(p=0.1)
  ↓
Output layer: 1 neuron (logits)
  → BCEWithLogitsLoss (with pos_weight for class imbalance)
```

**Total parameters:** ~14,000 (196×64 + 64×32 + biases)

**Design rationale:**
- Lightweight architecture for edge deployment
- GroupNorm(1) ensures DP compatibility (no cross-sample statistics)
- Minimal dropout to preserve learning capacity under DP noise

---

## 4.C Differential Privacy Implementation

### DP-SGD via Opacus

**Privacy parameters:**
- **Target δ:** 1e-5 (standard for DP-SGD, roughly 1/n)
- **Clipping norm (C):** 10.0 (increased from typical 1.0 for stability)
- **Privacy accountant:** Rényi Differential Privacy (RDP)
- **Noise multiplier (σ):** Computed by Opacus to achieve target ε

### Epsilon Budget (ε) Grid

**Phase 4 (Threat Analysis):**
- ε = 5.0 (Relaxed privacy)
- ε = 3.0 (Safe region)
- ε = ∞ (No DP baseline)

**Fairness Validation:**
- ε = 1.0, 3.0, 5.0, ∞

**Full experimental grid (Phase 5):**
- ε = {0.5, 1.0, 5.0, ∞}

### Adaptive Learning Rates (DP Mode)

When DP is enabled, learning rate is adjusted based on ε:
```python
if ε ≤ 1.0:   η = 0.05  (High noise → higher LR)
elif ε ≤ 5.0: η = 0.01  (Moderate noise)
else:          η = 0.005 (Low noise)
```

**Privacy budget accounting:**
- Total budget: ε × T × E (composition over rounds and epochs)
- Opacus tracks cumulative ε via RDP → (ε,δ)-DP conversion
- Reported ε values are **per-client, per-experiment** totals

---

## 4.D Attack Scenarios

### Malicious Client Assignment

**Deterministic selection:**
- Clients are indexed 0-9
- For attack_ratio = 0.2: clients [0, 1] are malicious
- For attack_ratio = 0.4: clients [0, 1, 2, 3] are malicious
- **Deterministic by index** (not random) for reproducibility

### Attack Types

#### 1. Label Flip Attack
- **Mechanism:** Malicious clients flip all labels (y → 1-y)
- **Goal:** Degrade global model accuracy
- **Implementation:** During local training, use `flipped_target = 1 - target`

#### 2. Model Poisoning Attack
- **Mechanism:** Amplify gradient updates by scaling factor λ
- **Scaling factor (λ):** **10.0** (default)
- **Formula:** `poisoned_params = original + λ × (trained - original)`
- **Goal:** Disproportionate influence on global aggregation
- **Reference:** Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning" (USENIX Security 2020)

### Robust Aggregation Strategies

| Aggregator | Parameters | Description |
|------------|------------|-------------|
| **FedAvg** | — | Weighted average by sample count |
| **Trimmed Mean** | γ = 0.1 | Trim 10% highest/lowest per coordinate |
| **Median** | — | Coordinate-wise median |
| **Krum** | f = ⌊attack_ratio × N⌋ | Select client with min distance to others |

**Trimmed Mean details:**
- Trim fraction (γ) = 0.1
- Trim count = ⌊γ × N⌋ = 1 client per side (for N=10)
- Trims 10% from both tails of parameter distribution

---

## 4.E Experimental Grid

### Phase 4: Threat Analysis (Attack Scenarios)

**Grid dimensions:**
- Epsilon: {3.0, 5.0, ∞}
- Aggregators: {FedAvg, Trimmed Mean, Median, Krum}
- Attacks: {none, label_flip, model_poisoning}
- Attack ratios: {0.0 (clean), 0.2, 0.4}

**Total configurations:**
```
3 epsilon × 4 aggregators × [
  1 (no attack) +
  2 attacks × 2 ratios
] = 3 × 4 × 5 = 60 configs
```

**Runs per config:** 3 independent runs
**Total experiments:** 60 × 3 = **180 experiments**

### Fairness Validation (Separate)

**Grid dimensions:**
- Epsilon: {1.0, 3.0, 5.0, ∞}
- Aggregator: Median only
- Attack: label_flip only
- Attack ratio: 0.4 only

**Total configurations:** 4
**Runs per config:** 1
**Total experiments:** 4

**Purpose:** Analyze per-attack-type detection rates under DP

---

## 4.F Evaluation Metrics

### Binary Classification Metrics
- **Accuracy:** Overall correctness
- **Precision:** True positives / (True positives + False positives)
- **Recall (TPR):** True positives / (True positives + False negatives)
- **F1 Score:** Harmonic mean of precision and recall
- **AUROC:** Area under ROC curve (if probabilities available)

### Fairness Metrics (Validation Only)
- **Detection Rate per Attack Category:** Recall for each attack type
- **Disparate Impact (DI):** min(DR) / max(DR) ≥ 0.8 (80% rule)
- **Performance Gap:** max(DR) - min(DR)
- **Common vs Rare Attack Comparison:**
  - Common: {Generic, Exploits, DoS}
  - Rare: {Analysis, Backdoor, Shellcode, Worms}

### Convergence Metrics
- **Convergence round:** First round where F1 ≥ 0.90
- **Best F1:** Maximum F1 across all rounds
- **Final F1:** F1 at last training round

---

## 4.G Computational Environment

**Platform:** Google Colab Pro
- **GPU:** NVIDIA Tesla T4 (16GB VRAM)
- **Memory:** 25GB RAM (Colab Pro)
- **Storage:** Google Drive mounted at `/content/drive/MyDrive/`

**Software Stack:**
- Python 3.10
- PyTorch 2.0+ (CUDA 11.8)
- Flower 1.5+ (FL framework)
- Opacus 1.4+ (DP library)
- scikit-learn 1.3+

**Resource allocation per client (simulation):**
- CPU cores: 1
- GPU share: 0.1 (10% of GPU per client)
- Enables parallel client simulation on single GPU

**Memory constraints:**
- Batch size limited to 256 to prevent OOM
- Gradient accumulation disabled
- Model checkpointing every 5 rounds

---

## 4.H Reproducibility

### Random Seed Control
- **Global seed:** 42
- Applied to: NumPy, PyTorch (CPU + CUDA)
- Ensures deterministic client assignment, data splits, weight initialization

### Checkpointing
- Results saved after each experiment completion
- JSON format with full config + metrics
- Allows resume on interruption

### Convergence Criteria
- Early stopping: F1 ≥ 0.90 for 10 consecutive rounds
- Maximum rounds: 50 (timeout)
- Prevents overfitting and reduces compute cost

---

## Summary Table for Paper

| **Parameter** | **Value** | **Rationale** |
|---------------|-----------|---------------|
| Dataset | UNSW-NB15 | Modern, realistic attacks |
| Samples | 175,341 (80/20 split) | Sufficient for FL |
| Features | 196 | Post-preprocessing |
| Clients (N) | 10 | Standard FL setup |
| FL Rounds (T) | 50 (max) | Early stopping enabled |
| Local Epochs (E) | 5 | Balance communication/compute |
| Batch Size (B) | 256 | Memory/convergence trade-off |
| Learning Rate (η) | 0.002 (base) | Adaptive for DP |
| Optimizer | SGD (mom=0.9) | Stable for FL |
| Model | MLP [196→64→32→1] | Lightweight, DP-compatible |
| DP Clipping (C) | 10.0 | High for stability |
| DP Delta (δ) | 1e-5 | Standard |
| Epsilon (ε) | {0.5, 1.0, 3.0, 5.0, ∞} | Privacy-utility spectrum |
| Trim Fraction (γ) | 0.1 | Trimmed Mean parameter |
| Attack Scale (λ) | 10.0 | Model poisoning factor |
| Attack Ratios | {0.0, 0.2, 0.4} | Clean + 20%/40% Byzantine |
| Independent Runs | 3 | Statistical significance |
| Total Experiments | 180+ (Phase 4) | Comprehensive grid |

---

## Key Differences: Phase 4 vs Fairness Validation

| Aspect | Phase 4 (Threat Analysis) | Fairness Validation |
|--------|---------------------------|---------------------|
| **Purpose** | Robustness under attacks | Per-attack-type fairness |
| **Epsilon values** | {0.3, 0.5, 1.0, 3.0, 5.0, ∞} | {1.0, 3.0, 5.0, ∞} |
| **Aggregators** | All 4 (FedAvg, TM, Median, Krum) | Median only |
| **Attacks** | label_flip + model_poisoning | label_flip only |
| **Attack ratios** | {0.2, 0.4} | 0.4 only |
| **Runs** | 3 | 1 |
| **Output** | Overall F1/accuracy | Per-category detection rates |
| **Experiments** | 180 | 4 |

---

## Answers to Your Specific Questions

1. **FL rounds (T):** 50 maximum (early stop if F1 ≥ 0.90 for 10 rounds)
   - Actual convergence: 4-47 rounds depending on config

2. **Local epochs:** 5 per round

3. **Model architecture:** MLP with hidden layers [64, 32]
   - Input: 196 features
   - Hidden 1: 64 neurons (GroupNorm + ReLU + Dropout 0.1)
   - Hidden 2: 32 neurons (GroupNorm + ReLU + Dropout 0.1)
   - Output: 1 neuron (binary classification)

4. **Optimizer and LR:**
   - SGD with momentum=0.9, weight_decay=1e-4
   - Base LR = 0.002 (no DP)
   - Adaptive LR for DP: 0.05 (ε≤1.0), 0.01 (ε≤5.0), 0.005 (ε>5.0)

5. **Batch size:** 256

6. **DP parameters:**
   - Clipping norm C = 10.0
   - Target delta δ = 1e-5
   - Noise multiplier σ = computed by Opacus

7. **Data partitioning:** IID split, random uniform sampling (not stratified)

8. **Trimmed Mean γ:** 0.1 ✓ (your assumption correct)

9. **Model poisoning λ:** 10.0 ✓ (your assumption correct)

---

## Next Steps for Section 4

1. **4.A Dataset:** Explain UNSW-NB15 choice + preprocessing (above table)
2. **4.B FL Config:** Report N=10, T=50, E=5, B=256, optimizer details
3. **4.C DP Implementation:** ε grid, C=10.0, δ=1e-5, adaptive LR
4. **4.D Experimental Grid:** 180 experiments (3 ε × 4 agg × 5 attack configs × 3 runs)
5. **4.E Evaluation:** F1/accuracy for threat analysis, per-category DR for fairness
6. **4.F Environment:** Colab T4 GPU, resource allocation

**Suggested subsection order:**
- 4.A: Dataset and Preprocessing (UNSW-NB15, 196 features, 80/20 split)
- 4.B: Federated Learning Configuration (N=10, T=50, E=5, IID partition)
- 4.C: Differential Privacy Implementation (Opacus, ε grid, adaptive LR)
- 4.D: Attack Scenarios (label flip + model poisoning, deterministic assignment)
- 4.E: Robust Aggregation (FedAvg, Trimmed Mean γ=0.1, Median, Krum)
- 4.F: Experimental Grid (180 experiments, 3 runs each)
- 4.G: Computational Environment (Colab T4, resource constraints)
