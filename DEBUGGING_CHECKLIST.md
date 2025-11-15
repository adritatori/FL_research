# FL-NIDS Training Failure - Debugging Checklist

## Problem Summary
Model outputs are constant (all 0s or all 1s), indicating no learning is occurring.
- Loss: constant at 0.6231
- AUC-ROC: 0.5 (random guessing)
- F1: alternates between 0.0 and 0.71

## Critical Checks (Run in Colab)

### 1. Verify Data Loading
Add this diagnostic code BEFORE running experiments:

```python
from main import UNSWDataLoader
from config import config
import numpy as np

# Load data
data_loader = UNSWDataLoader(config.DATA_PATH)
X, y, feature_names = data_loader.load_and_preprocess(use_sample=True, sample_fraction=0.01)

# Check data validity
print(f"Data shape: {X.shape}")
print(f"Label shape: {y.shape}")
print(f"Label distribution: 0s={np.sum(y==0)}, 1s={np.sum(y==1)}")
print(f"\nFeature statistics:")
print(f"  Min: {X.min()}")
print(f"  Max: {X.max()}")
print(f"  Mean: {X.mean()}")
print(f"  Std: {X.std()}")
print(f"  NaNs: {np.isnan(X).sum()}")
print(f"  Infs: {np.isinf(X).sum()}")
print(f"\nFirst 5 samples:")
print(X[:5])
print(f"\nFirst 5 labels:")
print(y[:5])
print(f"\nFeature variance (all features should have non-zero variance):")
feature_var = np.var(X, axis=0)
print(f"  Zero variance features: {np.sum(feature_var == 0)}")
print(f"  Min variance: {feature_var.min()}")
print(f"  Max variance: {feature_var.max()}")
```

**Expected:**
- Data should have non-zero variance across features
- No NaNs or Infs
- Labels should be roughly 45% class 0, 55% class 1

**If data looks wrong:** Check the dataset files in `/content/drive/MyDrive/IDSDatasets/UNSW 15/`

---

### 2. Test Model Training Without FL
Add this test to verify the model CAN learn:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from runner import IntrusionDetectionMLP
from config import config
from main import UNSWDataLoader

# Load small sample
data_loader = UNSWDataLoader(config.DATA_PATH)
X, y, _ = data_loader.load_and_preprocess(use_sample=True, sample_fraction=0.01)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create simple training setup
train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.FloatTensor(y_train).reshape(-1, 1)
)
trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Create model
model = IntrusionDetectionMLP(
    input_dim=X_train.shape[1],
    hidden_dims=[128, 64, 32],
    dropout=0.3
).to(config.DEVICE)

# Setup training
pos_weight = torch.tensor([np.sum(y_train==0) / np.sum(y_train==1)]).to(config.DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training simple model for 10 epochs...")
for epoch in range(10):
    model.train()
    total_loss = 0
    for data, target in trainloader:
        data, target = data.to(config.DEVICE), target.to(config.DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(X_test).to(config.DEVICE)
        logits = model(test_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(float)

    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"Epoch {epoch+1}: Loss={total_loss/len(trainloader):.4f}, F1={f1:.4f}, AUC={auc:.4f}")

print("\nIf F1 improves above 0.71 and AUC above 0.5, the model CAN learn!")
print("If it stays at F1=0.71 and AUC=0.5, there's a data or model issue.")
```

---

### 3. Check Flower Initial Parameters

The Flower strategy doesn't set initial_parameters! Add this fix to `runner.py`:

```python
# Around line 490, BEFORE creating the strategy:

# Initialize global model for Flower
init_model = IntrusionDetectionMLP(
    input_dim=input_dim,
    hidden_dims=base_config.HIDDEN_DIMS,
    dropout=base_config.DROPOUT_RATE
).to(base_config.DEVICE)
initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for _, val in init_model.state_dict().items()])

# Then add to strategy_params:
strategy_params = {
    "fraction_fit": base_config.CLIENT_FRACTION,
    "fraction_evaluate": 1.0,
    "min_fit_clients": base_config.MIN_FIT_CLIENTS,
    "min_available_clients": base_config.MIN_AVAILABLE_CLIENTS,
    "evaluate_metrics_aggregation_fn": weighted_avg,
    "fit_metrics_aggregation_fn": weighted_avg,
    "initial_parameters": initial_parameters,  # ADD THIS!
}
```

---

### 4. Add Gradient Monitoring

Add gradient checking to BaseClient.fit() to see if gradients are flowing:

```python
# In BaseClient.fit(), after the first loss.backward():
if epoch == 0 and num_batches == 0:
    grad_norm = 0
    for p in self.model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    print(f"  Client {self.cid} - First gradient norm: {grad_norm:.6f}")
    if grad_norm < 1e-7:
        print(f"  WARNING: Vanishing gradients!")
```

---

## Potential Fixes

### Fix #1: Initialize Flower with Global Model
Flower might not be initializing properly. Add `initial_parameters` to the strategy (see Check #3).

### Fix #2: Adjust Learning Rate
Try increasing the learning rate:
```python
# In config.py
LEARNING_RATE = 0.01  # Instead of 0.001
```

### Fix #3: Simplify Model Architecture
GroupNorm might be causing issues. Try BatchNorm:

```python
# In IntrusionDetectionMLP, replace GroupNorm with BatchNorm:
layers.extend([
    nn.Linear(prev_dim, hidden_dim),
    nn.BatchNorm1d(hidden_dim),  # Instead of GroupNorm
    nn.ReLU(),
    nn.Dropout(dropout)
])
```

### Fix #4: Check Dataset Files
Verify the UNSW-NB15 files exist and are correct:
```bash
!ls -lh "/content/drive/MyDrive/IDSDatasets/UNSW 15/"
```

---

## Next Steps

1. Run Check #1 first - if data is broken, nothing else matters
2. Run Check #2 - if simple training works, the issue is in FL setup
3. If simple training fails, the issue is data/model/hyperparameters
4. If simple training works, apply Fix #1 (initial_parameters)

Report back with results and I'll help further!
