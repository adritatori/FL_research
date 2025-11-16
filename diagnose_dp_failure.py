import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np
import sys
from pathlib import Path
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Import from existing code
from runner import IntrusionDetectionMLP
from main import UNSWDataLoader
from config import config


def test_dp_training(
    X_train,
    y_train,
    X_test,
    y_test,
    epsilon: float = 5.0,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    max_grad_norm: float = 10.0,
    num_epochs: int = 10,
    verbose: bool = True,
):
    """Test single-client DP training with specific hyperparameters."""

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).reshape(-1, 1),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = IntrusionDetectionMLP(
        input_dim=X_train.shape[1],
        hidden_dims=[64, 32],
        dropout=0.1,
    ).to(config.DEVICE)

    # Setup optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
    )

    # Setup loss with class weights
    pos_weight = torch.tensor([np.sum(y_train == 0) / np.sum(y_train == 1)]).to(
        config.DEVICE
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Setup differential privacy
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=epsilon,
        target_delta=1e-5,
        epochs=num_epochs,
        max_grad_norm=max_grad_norm,
    )

    if verbose:
        print("\nTesting Configuration:")
        print(f"  Epsilon: {epsilon}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Max Grad Norm: {max_grad_norm}")
        print(f"  Pos Weight: {pos_weight.item():.3f}")
        print(f"  Num Epochs: {num_epochs}")

    # Training loop
    results = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Track gradients
        total_grad_norm = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            if torch.isnan(loss):
                print(f"  [ERROR] NaN loss at epoch {epoch + 1}, batch {batch_idx}")
                continue

            loss.backward()

            # Measure gradient norm BEFORE optimizer step (after DP clipping)
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            total_grad_norm += grad_norm

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(X_test).to(config.DEVICE)
            logits = model(test_tensor).cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))  # sigmoid
            probs_flat = probs.ravel()
            preds = (probs > 0.5).astype(float).ravel()

        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs_flat)

        # Check epsilon consumed
        epsilon_consumed = privacy_engine.get_epsilon(delta=1e-5)

        results.append(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "grad_norm": avg_grad_norm,
                "accuracy": accuracy,
                "f1": f1,
                "auc": auc,
                "epsilon": epsilon_consumed,
            }
        )

        if verbose:
            print(
                f"Epoch {epoch + 1:2d}: "
                f"Loss={avg_loss:.4f}, "
                f"GradNorm={avg_grad_norm:.4f}, "
                f"Acc={accuracy:.4f}, "
                f"F1={f1:.4f}, "
                f"AUC={auc:.4f}, "
                f"ε={epsilon_consumed:.2f}"
            )

    return results


def main():
    print("=" * 80)
    print("DP TRAINING FAILURE DIAGNOSTIC")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading data...")
    data_loader = UNSWDataLoader(config.DATA_PATH)
    X, y, feature_names = data_loader.load_and_preprocess(
        use_sample=True,
        sample_fraction=0.01,  # Use 1% for fast testing
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Class distribution: {np.sum(y_train == 0)} / {np.sum(y_train == 1)} (0s/1s)")
    print(f"  Feature dimension: {X_train.shape[1]}")

    # Data sanity checks
    print("\n[2/5] Data sanity checks...")
    print(f"  NaNs in X_train: {np.isnan(X_train).sum()}")
    print(f"  Infs in X_train: {np.isinf(X_train).sum()}")
    print(f"  X_train range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"  X_train mean: {X_train.mean():.3f}, std: {X_train.std():.3f}")

    zero_var_features = np.var(X_train, axis=0) == 0
    print(f"  Zero-variance features: {np.sum(zero_var_features)}")

    # Test 1: Baseline - Current failing configuration
    print("\n[3/5] TEST 1: Current Configuration (FAILING)")
    print("-" * 80)
    results_baseline = test_dp_training(
        X_train,
        y_train,
        X_test,
        y_test,
        epsilon=5.0,
        batch_size=256,  # Current
        learning_rate=0.01,  # Current
        max_grad_norm=10.0,
        num_epochs=10,
    )

    # Test 2: Smaller batch size
    print("\n[4/5] TEST 2: Smaller Batch Size (64 vs 256)")
    print("-" * 80)
    results_small_batch = test_dp_training(
        X_train,
        y_train,
        X_test,
        y_test,
        epsilon=5.0,
        batch_size=64,  # CHANGED
        learning_rate=0.01,
        max_grad_norm=10.0,
        num_epochs=10,
    )

    # Test 3: Higher learning rate
    print("\n[5/5] TEST 3: Higher Learning Rate + Small Batch (0.05 vs 0.01)")
    print("-" * 80)
    results_high_lr = test_dp_training(
        X_train,
        y_train,
        X_test,
        y_test,
        epsilon=5.0,
        batch_size=64,
        learning_rate=0.05,  # CHANGED
        max_grad_norm=10.0,
        num_epochs=10,
    )

    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    def summarize(results, name: str) -> None:
        final = results[-1]
        best_f1 = max(r["f1"] for r in results)
        print(f"\n{name}:")
        print(
            f"  Final: Acc={final['accuracy']:.4f}, "
            f"F1={final['f1']:.4f}, AUC={final['auc']:.4f}"
        )
        print(f"  Best F1: {best_f1:.4f}")
        print(f"  Final Grad Norm: {final['grad_norm']:.4f}")
        print(f"  Loss change: {results[0]['loss']:.4f} → {final['loss']:.4f}")

        # Check if learning occurred
        if final["auc"] > 0.55 and best_f1 > 0.3:
            print("  ✅ LEARNING OCCURRED")
        elif final["auc"] > 0.52 or final["f1"] > 0.1:
            print("  ⚠️  WEAK LEARNING")
        else:
            print("  ❌ NO LEARNING (random)")

    summarize(results_baseline, "TEST 1 - Current Config (BS=256, LR=0.01)")
    summarize(results_small_batch, "TEST 2 - Small Batch (BS=64, LR=0.01)")
    summarize(results_high_lr, "TEST 3 - Small Batch + High LR (BS=64, LR=0.05)")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    best_f1_baseline = max(r["f1"] for r in results_baseline)
    best_f1_small = max(r["f1"] for r in results_small_batch)
    best_f1_high = max(r["f1"] for r in results_high_lr)

    if best_f1_high > max(best_f1_baseline, best_f1_small) + 0.1:
        print("\n✅ FIX: Use BATCH_SIZE=64 and LR=0.05 for epsilon=5.0")
        print("   Update config.py and runner.py accordingly")
    elif best_f1_small > best_f1_baseline + 0.1:
        print("\n✅ FIX: Use BATCH_SIZE=64 (keep LR=0.01)")
        print("   Update config.py: BATCH_SIZE = 64")
    elif best_f1_baseline > 0.3:
        print("\n⚠️  Current config works but suboptimal")
        print("   Consider tuning further")
    else:
        print("\n❌ CRITICAL: None of the configs work!")
        print("   Deeper investigation needed:")
        print("   - Check data preprocessing")
        print("   - Test without DP first")
        print("   - Verify Opacus installation")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
