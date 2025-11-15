"""
Debug script to diagnose why the model is not learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from runner import IntrusionDetectionMLP, BaseClient
from config import ExperimentConfig
from main import UNSWDataLoader


print("=" * 80)
print("DIAGNOSTIC: Model Training Failure Investigation")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Load config and data
# ---------------------------------------------------------------------------

config = ExperimentConfig()
config.set_seed(42)

print("\n1. Loading data...")
data_loader = UNSWDataLoader(config.DATA_PATH)
X, y, _ = data_loader.load_and_preprocess(use_sample=True, sample_fraction=0.01)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"   Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"   Train labels - 0s: {np.sum(y_train == 0)}, 1s: {np.sum(y_train == 1)}")

# Build datasets
train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.FloatTensor(y_train).view(-1, 1)
)

test_dataset = TensorDataset(
    torch.FloatTensor(X_test),
    torch.FloatTensor(y_test).view(-1, 1)
)

trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=128)

print(f"   Trainloader batches: {len(trainloader)}")
print(f"   Testloader batches: {len(testloader)}")

# ---------------------------------------------------------------------------
# 2. Model creation
# ---------------------------------------------------------------------------

print("\n2. Creating model...")

model = IntrusionDetectionMLP(
    input_dim=X_train.shape[1],
    hidden_dims=[128, 64, 32],
    dropout=0.3
).to(config.DEVICE)

print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Model device: {next(model.parameters()).device}")

# Loss + optimizer
num_class_0 = np.sum(y_train == 0)
num_class_1 = np.sum(y_train == 1)
pos_weight = torch.tensor([num_class_0 / num_class_1]).to(config.DEVICE)

print(f"   Pos weight: {pos_weight.item():.4f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------------------------------------------
# 3. Initial model predictions
# ---------------------------------------------------------------------------

print("\n3. Testing initial model predictions...")
model.eval()

with torch.no_grad():
    for data, target in testloader:
        data = data.to(config.DEVICE)
        logits = model(data)
        probs = torch.sigmoid(logits)

        print(f"   Logits - min: {logits.min().item():.4f}, "
              f"max: {logits.max().item():.4f}, mean: {logits.mean().item():.4f}")

        print(f"   Probs  - min: {probs.min().item():.4f}, "
              f"max: {probs.max().item():.4f}, mean: {probs.mean().item():.4f}")

        break

# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------

print("\n4. Training for 10 epochs...")
model.train()

for epoch in range(10):
    total_loss = 0
    num_batches = 0

    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(config.DEVICE), target.to(config.DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if torch.isnan(loss):
            print("   ERROR: NaN loss detected!")
            break

        loss.backward()

        # Gradient norm check
        if epoch == 0 and batch_idx == 0:
            total_grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            print(f"   First batch gradient norm: {total_grad_norm:.6f}")

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches

    # Evaluate
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            logits = model(data)
            preds = (torch.sigmoid(logits) > 0.5).float()

            correct += (preds == target).sum().item()
            total += target.size(0)

    accuracy = correct / total
    print(f"   Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

    model.train()

# ---------------------------------------------------------------------------
# 5. Final evaluation
# ---------------------------------------------------------------------------

print("\n5. Final model predictions...")
model.eval()

all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for data, target in testloader:
        data = data.to(config.DEVICE)
        logits = model(data)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        all_probs.extend(probs.cpu().numpy().flatten())
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(target.numpy().flatten())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

from sklearn.metrics import f1_score, roc_auc_score

f1 = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)

print(f"   Final F1: {f1:.4f}")
print(f"   Final AUC-ROC: {auc:.4f}")
print(f"   Prediction distribution: 0s={np.sum(all_preds == 0)}, "
      f"1s={np.sum(all_preds == 1)}")
print(f"   Label distribution: 0s={np.sum(all_labels == 0)}, "
      f"1s={np.sum(all_labels == 1)}")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
