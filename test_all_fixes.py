import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from opacus import PrivacyEngine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

import numpy as np
from datetime import datetime

from runner import IntrusionDetectionMLP
from main import UNSWDataLoader
from config import config


def test_configuration(
    X_train,
    y_train,
    X_test,
    y_test,
    test_name,
    epsilon: float = 5.0,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    max_grad_norm: float = 10.0,
    local_epochs: int = 5,
    verbose: bool = False,
):
    """Test a specific configuration."""

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).reshape(-1, 1),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create model (already uses GroupNorm)
    model = IntrusionDetectionMLP(
        input_dim=X_train.shape[1],
        hidden_dims=[64, 32],
        dropout=0.1,
    ).to(config.DEVICE)

    # Setup optimizer with SGD (as in production)
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

    # Setup differential privacy if needed
    use_dp = epsilon != float("inf")
    if use_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=epsilon,
            target_delta=1e-5,
            epochs=local_epochs,
            max_grad_norm=max_grad_norm,
        )

    print("\n" + "=" * 80)
    print(test_name)
    print("=" * 80)
    print(f"  Epsilon: {epsilon} {'(NO DP)' if epsilon == float('inf') else ''}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Max Grad Norm: {max_grad_norm}")
    print(f"  Local Epochs: {local_epochs}")
    print(f"  Pos Weight: {pos_weight.item():.3f}")
    print(f"  Using DP: {use_dp}")
    print("-" * 80)

    # Training loop
    for epoch in range(local_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            if torch.isnan(loss):
                print(f"  [ERROR] NaN loss at epoch {epoch + 1}, batch {batch_idx}")
                continue

            loss.backward()

            # Manual clipping only if no DP (Opacus handles it otherwise)
            if not use_dp:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(X_test).to(config.DEVICE)
            logits = model(test_tensor).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            probs_flat = probs.ravel()
            preds = (probs > 0.5).astype(float).ravel()

        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs_flat)

        if verbose or (epoch + 1) % 2 == 0:
            print(
                f"  Epoch {epoch + 1:2d}/{local_epochs}: "
                f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}, "
                f"F1={f1:.4f}, AUC={auc:.4f}"
            )

    # Final evaluation
    print("\n  FINAL RESULTS:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    F1 Score: {f1:.4f}")
    print(f"    AUC-ROC:  {auc:.4f}")

    # Interpret results
    if auc > 0.60 and f1 > 0.4:
        print("    Status: ✅ GOOD - Model is learning well")
        status = "GOOD"
    elif auc > 0.55 and f1 > 0.2:
        print("    Status: ⚠️  WEAK - Some learning, but suboptimal")
        status = "WEAK"
    elif auc > 0.52 or f1 > 0.1:
        print("    Status: ❌ POOR - Minimal learning")
        status = "POOR"
    else:
        print("    Status: ❌ FAILED - No learning (random guessing)")
        status = "FAILED"

    return {
        "test_name": test_name,
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc,
        "status": status,
        "config": {
            "epsilon": epsilon,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "local_epochs": local_epochs,
        },
    }


def main():
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DP TRAINING FIX TESTING")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\n[SETUP] Loading data...")
    data_loader = UNSWDataLoader(config.DATA_PATH)
    X, y, feature_names = data_loader.load_and_preprocess(
        use_sample=True,
        sample_fraction=0.01,  # 1% for fast testing
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Class split: {np.sum(y_train == 0)}/{np.sum(y_train == 1)} (0s/1s)")

    results = []

    # TEST 0: Baseline (current failing config)
    results.append(
        test_configuration(
            X_train,
            y_train,
            X_test,
            y_test,
            test_name="TEST 0: BASELINE (Current Failing Config)",
            epsilon=5.0,
            batch_size=256,
            learning_rate=0.01,
            local_epochs=5,
        )
    )

    # TEST 1: No DP (epsilon=inf) - Confirms DP is the issue
    results.append(
        test_configuration(
            X_train,
            y_train,
            X_test,
            y_test,
            test_name="TEST 1: NO DP (epsilon=∞) - Baseline without privacy",
            epsilon=float("inf"),
            batch_size=256,
            learning_rate=0.002,  # Base LR without DP
            local_epochs=5,
        )
    )

    # TEST 2: Lower LR (user suggestion)
    results.append(
        test_configuration(
            X_train,
            y_train,
            X_test,
            y_test,
            test_name="TEST 2: LOWER LR (User Suggestion)",
            epsilon=5.0,
            batch_size=256,
            learning_rate=0.0005,  # User's suggestion
            local_epochs=5,
        )
    )

    # TEST 3: Higher LR (DP-SGD theory)
    results.append(
        test_configuration(
            X_train,
            y_train,
            X_test,
            y_test,
            test_name="TEST 3: HIGHER LR (DP-SGD Theory)",
            epsilon=5.0,
            batch_size=256,
            learning_rate=0.05,  # 25x base LR
            local_epochs=5,
        )
    )

    # TEST 4: Smaller batch size
    results.append(
        test_configuration(
            X_train,
            y_train,
            X_test,
            y_test,
            test_name="TEST 4: SMALLER BATCH SIZE",
            epsilon=5.0,
            batch_size=64,
            learning_rate=0.01,
            local_epochs=5,
        )
    )

    # TEST 5: More local epochs (user suggestion)
    results.append(
        test_configuration(
            X_train,
            y_train,
            X_test,
            y_test,
            test_name="TEST 5: MORE LOCAL EPOCHS (User Suggestion)",
            epsilon=5.0,
            batch_size=256,
            learning_rate=0.01,
            local_epochs=10,
        )
    )

    # TEST 6: Combined - Small batch + High LR + More epochs
    results.append(
        test_configuration(
            X_train,
            y_train,
            X_test,
            y_test,
            test_name="TEST 6: COMBINED OPTIMAL (Small BS + High LR + More Epochs)",
            epsilon=5.0,
            batch_size=64,
            learning_rate=0.05,
            local_epochs=10,
        )
    )

    # Summary
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"\n{'Test':<50} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Status':>10}")
    print("-" * 80)

    for r in results:
        print(
            f"{r['test_name'][:50]:<50} "
            f"{r['accuracy']:>8.4f} {r['f1']:>8.4f} {r['auc']:>8.4f} "
            f"{r['status']:>10}"
        )

    # Find best result
    best = max(results[1:], key=lambda x: x["f1"])  # Skip baseline

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print(f"\n🏆 BEST CONFIGURATION: {best['test_name']}")
    print(f"   Accuracy: {best['accuracy']:.4f}")
    print(f"   F1 Score: {best['f1']:.4f}")
    print(f"   AUC-ROC:  {best['auc']:.4f}")
    print("\n   Config:")
    print(f"     BATCH_SIZE = {best['config']['batch_size']}")
    print(f"     LEARNING_RATE = {best['config']['learning_rate']}")
    print(f"     LOCAL_EPOCHS = {best['config']['local_epochs']}")

    # Check if no DP works
    no_dp_result = results[1]
    if no_dp_result["status"] == "GOOD" and best["status"] != "GOOD":
        print("\n⚠️  WARNING: No-DP works well but DP configs fail.")
        print("   This confirms DP is causing the issue.")
        print("   You may need to accept lower accuracy with privacy,")
        print("   or use the best DP config above as a compromise.")

    if best["config"]["learning_rate"] > 0.01:
        print(
            f"\n💡 INSIGHT: Higher LR ({best['config']['learning_rate']}) works better with DP"
        )
        print("   This confirms DP-SGD theory: noise requires stronger learning signal")
    elif best["config"]["learning_rate"] < 0.01:
        print(
            f"\n💡 INSIGHT: Lower LR ({best['config']['learning_rate']}) works better"
        )
        print("   This suggests gradient instability, not signal weakness")

    if best["config"]["batch_size"] < 256:
        print(
            f"\n💡 INSIGHT: Smaller batch size ({best['config']['batch_size']}) helps"
        )
        print("   Opacus per-sample gradients work better with smaller batches")

    print("\n" + "=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
