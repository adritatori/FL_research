"""
NEURAL NETWORK BUG ISOLATION
==============================
CRITICAL FINDING: sklearn achieves AUC=0.9632 but our NN gets AUC=0.5

This script isolates the EXACT bug in the neural network.

Tests:
1. Simple MLP WITHOUT GroupNorm (just Linear + ReLU)
2. Current architecture WITHOUT pos_weight
3. Current architecture with different weight init
4. Gradient flow monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np

from runner import IntrusionDetectionMLP
from main import UNSWDataLoader
from config import config


class SimpleMLP(nn.Module):
    """Simple MLP WITHOUT GroupNorm - just to test if GroupNorm is the issue"""
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SimpleMLP_BatchNorm(nn.Module):
    """Simple MLP WITH BatchNorm (not DP-safe but testing)"""
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def test_network(model, X_train, y_train, X_test, y_test, test_name,
                 use_pos_weight=True, lr=0.001, epochs=30, batch_size=128):
    """Test a specific network architecture"""

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).reshape(-1, 1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss function
    if use_pos_weight:
        pos_weight = torch.tensor([np.sum(y_train==0) / np.sum(y_train==1)]).to(config.DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        pw_str = f"{pos_weight.item():.3f}"
    else:
        criterion = nn.BCEWithLogitsLoss()
        pw_str = "None"

    print(f"\n{'='*80}")
    print(f"{test_name}")
    print(f"{'='*80}")
    print(f"  Pos Weight: {pw_str}")
    print(f"  Learning Rate: {lr}")
    print(f"  Epochs: {epochs}")

    # Check initial gradient flow
    model.train()
    data, target = next(iter(train_loader))
    data, target = data.to(config.DEVICE), target.to(config.DEVICE)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append((name, param.grad.norm().item()))

    print(f"\n  Initial Gradient Norms:")
    for name, norm in grad_norms[:5]:
        print(f"    {name}: {norm:.6f}")

    total_grad = sum(n for _, n in grad_norms)
    print(f"    TOTAL: {total_grad:.6f}")

    if total_grad < 1e-6:
        print(f"    ⚠️  WARNING: Gradients are near ZERO!")

    print(f"\n  Training Progress:")

    best_auc = 0.5

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for data, target in train_loader:
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(X_test).to(config.DEVICE)
            logits = model(test_tensor).cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))
            preds = (probs > 0.5).astype(float).flatten()

        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        best_auc = max(best_auc, auc)

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}, "
                  f"F1={f1:.4f}, AUC={auc:.4f}")

    print(f"\n  FINAL: AUC={auc:.4f} (Best={best_auc:.4f})")

    if best_auc > 0.60:
        print(f"  ✅ NETWORK CAN LEARN!")
    else:
        print(f"  ❌ NETWORK CANNOT LEARN")

    return {'auc': auc, 'best_auc': best_auc, 'f1': f1}


def main():
    print("\n" + "="*80)
    print("NEURAL NETWORK BUG ISOLATION")
    print("="*80)
    print("Target: sklearn AUC=0.9632, Current NN AUC=0.5000")
    print("Testing different architectures to isolate the bug...")

    # Load data (10% sample for faster testing)
    from main import UNSWDataLoader
    data_loader = UNSWDataLoader(config.DATA_PATH)
    X, y, _ = data_loader.load_and_preprocess(use_sample=True, sample_fraction=0.10)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\nData: {len(X_train)} train, {len(X_test)} test, {X_train.shape[1]} features")
    print(f"Class balance: {np.sum(y_train==0)}/{np.sum(y_train==1)} (0s/1s)")

    results = {}

    # TEST 1: Simple MLP without GroupNorm
    model1 = SimpleMLP(
        input_dim=X_train.shape[1],
        hidden_dims=[64, 32],
        dropout=0.1
    ).to(config.DEVICE)
    results['simple_mlp'] = test_network(
        model1, X_train, y_train, X_test, y_test,
        "TEST 1: Simple MLP (No GroupNorm, No BatchNorm)",
        use_pos_weight=True
    )

    # TEST 2: Simple MLP without pos_weight
    model2 = SimpleMLP(
        input_dim=X_train.shape[1],
        hidden_dims=[64, 32],
        dropout=0.1
    ).to(config.DEVICE)
    results['simple_no_pw'] = test_network(
        model2, X_train, y_train, X_test, y_test,
        "TEST 2: Simple MLP WITHOUT pos_weight",
        use_pos_weight=False
    )

    # TEST 3: Current architecture (with GroupNorm)
    model3 = IntrusionDetectionMLP(
        input_dim=X_train.shape[1],
        hidden_dims=[64, 32],
        dropout=0.1
    ).to(config.DEVICE)
    results['groupnorm'] = test_network(
        model3, X_train, y_train, X_test, y_test,
        "TEST 3: Current Architecture (GroupNorm)",
        use_pos_weight=True
    )

    # TEST 4: Current architecture WITHOUT pos_weight
    model4 = IntrusionDetectionMLP(
        input_dim=X_train.shape[1],
        hidden_dims=[64, 32],
        dropout=0.1
    ).to(config.DEVICE)
    results['groupnorm_no_pw'] = test_network(
        model4, X_train, y_train, X_test, y_test,
        "TEST 4: GroupNorm WITHOUT pos_weight",
        use_pos_weight=False
    )

    # TEST 5: With BatchNorm (not DP-safe but diagnostic)
    model5 = SimpleMLP_BatchNorm(
        input_dim=X_train.shape[1],
        hidden_dims=[64, 32],
        dropout=0.1
    ).to(config.DEVICE)
    results['batchnorm'] = test_network(
        model5, X_train, y_train, X_test, y_test,
        "TEST 5: BatchNorm (for comparison)",
        use_pos_weight=True
    )

    # TEST 6: Larger network
    model6 = SimpleMLP(
        input_dim=X_train.shape[1],
        hidden_dims=[128, 64, 32],
        dropout=0.1
    ).to(config.DEVICE)
    results['larger'] = test_network(
        model6, X_train, y_train, X_test, y_test,
        "TEST 6: Larger Simple MLP [128, 64, 32]",
        use_pos_weight=True
    )

    # TEST 7: Xavier initialization
    model7 = SimpleMLP(
        input_dim=X_train.shape[1],
        hidden_dims=[64, 32],
        dropout=0.1
    ).to(config.DEVICE)
    for m in model7.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    results['xavier'] = test_network(
        model7, X_train, y_train, X_test, y_test,
        "TEST 7: Simple MLP with Xavier Init",
        use_pos_weight=True
    )

    # Summary
    print("\n" + "="*80)
    print("BUG ISOLATION SUMMARY")
    print("="*80)
    print(f"\n{'Test':<50} {'AUC':>10} {'Best AUC':>10} {'Status':>10}")
    print("-"*80)

    for name, res in results.items():
        status = "✅ WORKS" if res['best_auc'] > 0.60 else "❌ FAILS"
        print(f"{name:<50} {res['auc']:>10.4f} {res['best_auc']:>10.4f} {status:>10}")

    # Identify the bug
    print("\n" + "="*80)
    print("ROOT CAUSE IDENTIFICATION")
    print("="*80)

    if results['simple_mlp']['best_auc'] > 0.60:
        if results['groupnorm']['best_auc'] < 0.55:
            print("\n🎯 BUG FOUND: GroupNorm is breaking the model!")
            print("   SOLUTION: Remove GroupNorm or use LayerNorm instead")
        elif results['simple_no_pw']['best_auc'] > 0.60 and results['groupnorm']['best_auc'] < 0.55:
            print("\n🎯 BUG FOUND: Combination of GroupNorm + pos_weight is problematic")
            print("   SOLUTION: Either remove GroupNorm or remove pos_weight")
        else:
            print("\n✅ Simple MLP works, need to check what's different in production")
    else:
        print("\n❌ Even simple MLP doesn't work - deeper architecture issue")
        print("   Check: Learning rate, weight initialization, or data preprocessing")

    if results['batchnorm']['best_auc'] > 0.60 and results['groupnorm']['best_auc'] < 0.55:
        print("\n💡 BatchNorm works but GroupNorm doesn't")
        print("   This confirms GroupNorm is the issue")
        print("   SOLUTION: Use nn.GroupNorm(1, hidden_dim) (LayerNorm equivalent)")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
