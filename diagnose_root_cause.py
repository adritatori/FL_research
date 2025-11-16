"""
ROOT CAUSE DIAGNOSIS: Model Doesn't Learn Even WITHOUT DP!
============================================================
The previous tests revealed that EVEN WITHOUT DIFFERENTIAL PRIVACY,
the model shows AUC=0.5 (random guessing). This means:

1. DP is NOT the problem
2. The model itself cannot learn from the data
3. Likely issues: sample size too small (658 samples), optimizer choice, or data issues

This script tests the REAL root causes:
1. Sample size: 1% vs 10% of data
2. Optimizer: SGD vs Adam
3. Training duration: More epochs
4. Data learnability: Can ANY model learn from this data?
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from datetime import datetime

from runner import IntrusionDetectionMLP
from main import UNSWDataLoader
from config import config


def test_model(X_train, y_train, X_test, y_test, test_name, optimizer_type='sgd',
               lr=0.01, epochs=20, batch_size=64, verbose=True):
    """Test training without DP to isolate model learning issues"""

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).reshape(-1, 1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = IntrusionDetectionMLP(
        input_dim=X_train.shape[1],
        hidden_dims=[64, 32],
        dropout=0.1
    ).to(config.DEVICE)

    # Setup optimizer
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else:  # sgd
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # Loss with class weighting
    pos_weight = torch.tensor([np.sum(y_train==0) / np.sum(y_train==1)]).to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if verbose:
        print(f"\n{'='*80}")
        print(f"{test_name}")
        print(f"{'='*80}")
        print(f"  Train samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Optimizer: {optimizer_type.upper()}")
        print(f"  Learning Rate: {lr}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Batches per epoch: {len(train_loader)}")
        print(f"{'-'*80}")

    best_f1 = 0
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
            preds = (probs > 0.5).astype(float)

        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        best_f1 = max(best_f1, f1)
        best_auc = max(best_auc, auc)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: Loss={avg_loss:.4f}, "
                  f"Acc={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    # Final results
    if verbose:
        print(f"\n  FINAL RESULTS:")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    F1 Score: {f1:.4f} (Best: {best_f1:.4f})")
        print(f"    AUC-ROC:  {auc:.4f} (Best: {best_auc:.4f})")

        if best_auc > 0.60:
            print(f"    Status: ✅ LEARNING - Model discriminates between classes")
        elif best_auc > 0.52:
            print(f"    Status: ⚠️  WEAK - Slight learning detected")
        else:
            print(f"    Status: ❌ NO LEARNING - Random guessing")

    return {
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
        'best_f1': best_f1,
        'best_auc': best_auc
    }


def test_sklearn_baseline(X_train, y_train, X_test, y_test):
    """Test if data is learnable at all using sklearn"""
    print(f"\n{'='*80}")
    print("SKLEARN BASELINE - Is data learnable at all?")
    print(f"{'='*80}")

    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train, y_train)

    preds = lr.predict(X_test)
    probs = lr.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"  Logistic Regression:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    F1 Score: {f1:.4f}")
    print(f"    AUC-ROC:  {auc:.4f}")

    if auc > 0.60:
        print(f"    ✅ DATA IS LEARNABLE - Problem is in neural network training")
    elif auc > 0.55:
        print(f"    ⚠️  DATA PARTIALLY LEARNABLE - Features have weak signal")
    else:
        print(f"    ❌ DATA NOT LEARNABLE - Features don't contain useful information")

    return {'accuracy': accuracy, 'f1': f1, 'auc': auc}


def main():
    print("\n" + "="*80)
    print("ROOT CAUSE DIAGNOSIS: WHY ISN'T THE MODEL LEARNING?")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Test 1: Small sample (1%) - current failing configuration
    print("\n[TEST 1] SMALL DATA SAMPLE (1% = ~823 samples)")
    print("-" * 80)

    data_loader = UNSWDataLoader(config.DATA_PATH)
    X_small, y_small, _ = data_loader.load_and_preprocess(
        use_sample=True,
        sample_fraction=0.01  # 1% - current config
    )

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_small, y_small, test_size=0.2, random_state=42, stratify=y_small
    )

    scaler_s = StandardScaler()
    X_train_s = scaler_s.fit_transform(X_train_s)
    X_test_s = scaler_s.transform(X_test_s)

    # Sklearn baseline on small data
    sklearn_small = test_sklearn_baseline(X_train_s, y_train_s, X_test_s, y_test_s)

    # Neural network with Adam (better for small data)
    results_small_adam = test_model(
        X_train_s, y_train_s, X_test_s, y_test_s,
        test_name="Small Data (1%) + Adam Optimizer",
        optimizer_type='adam',
        lr=0.001,
        epochs=50,  # More epochs for small data
        batch_size=32  # Smaller batch
    )

    # Test 2: Larger sample (10%)
    print("\n[TEST 2] LARGER DATA SAMPLE (10% = ~8,233 samples)")
    print("-" * 80)

    X_large, y_large, _ = data_loader.load_and_preprocess(
        use_sample=True,
        sample_fraction=0.10  # 10% - much more data
    )

    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        X_large, y_large, test_size=0.2, random_state=42, stratify=y_large
    )

    scaler_l = StandardScaler()
    X_train_l = scaler_l.fit_transform(X_train_l)
    X_test_l = scaler_l.transform(X_test_l)

    # Sklearn baseline on larger data
    sklearn_large = test_sklearn_baseline(X_train_l, y_train_l, X_test_l, y_test_l)

    # Neural network with Adam on larger data
    results_large_adam = test_model(
        X_train_l, y_train_l, X_test_l, y_test_l,
        test_name="Larger Data (10%) + Adam Optimizer",
        optimizer_type='adam',
        lr=0.001,
        epochs=30,
        batch_size=128
    )

    # Neural network with SGD on larger data (production config)
    results_large_sgd = test_model(
        X_train_l, y_train_l, X_test_l, y_test_l,
        test_name="Larger Data (10%) + SGD Optimizer (Production)",
        optimizer_type='sgd',
        lr=0.002,
        epochs=30,
        batch_size=256
    )

    # Summary
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)

    print(f"\n{'Configuration':<45} {'Samples':>10} {'AUC':>8} {'F1':>8} {'Status':>15}")
    print("-" * 95)
    print(f"{'Sklearn (1% data)':<45} {len(X_train_s):>10} {sklearn_small['auc']:>8.4f} {sklearn_small['f1']:>8.4f}")
    print(f"{'NN + Adam (1% data)':<45} {len(X_train_s):>10} {results_small_adam['best_auc']:>8.4f} {results_small_adam['best_f1']:>8.4f}")
    print(f"{'Sklearn (10% data)':<45} {len(X_train_l):>10} {sklearn_large['auc']:>8.4f} {sklearn_large['f1']:>8.4f}")
    print(f"{'NN + Adam (10% data)':<45} {len(X_train_l):>10} {results_large_adam['best_auc']:>8.4f} {results_large_adam['best_f1']:>8.4f}")
    print(f"{'NN + SGD (10% data) [Production]':<45} {len(X_train_l):>10} {results_large_sgd['best_auc']:>8.4f} {results_large_sgd['best_f1']:>8.4f}")

    print("\n" + "="*80)
    print("ROOT CAUSE IDENTIFIED")
    print("="*80)

    # Determine root cause
    if sklearn_small['auc'] > 0.60:
        print("\n✅ Data IS learnable with sklearn on small sample")
        if results_small_adam['best_auc'] < 0.55:
            print("❌ Neural network fails to learn → Problem is NN training configuration")
            print("\n   RECOMMENDED FIXES:")
            print("   1. Use Adam optimizer instead of SGD for small data")
            print("   2. Reduce learning rate to 0.001 or 0.0005")
            print("   3. Increase training epochs to 50+")
            print("   4. Use smaller batch size (32-64)")
        else:
            print("✅ Neural network learns with Adam → Problem was SGD optimizer")
            print("\n   RECOMMENDED FIX:")
            print("   Switch to Adam optimizer in runner.py")
    else:
        print("\n⚠️  Data is hard to learn even with sklearn on small sample")

        if sklearn_large['auc'] > 0.60:
            print("✅ BUT larger sample (10%) is learnable")
            print("\n   ROOT CAUSE: Sample size too small (1%)")
            print("\n   RECOMMENDED FIX:")
            print("   In config.py, use sample_fraction=0.1 or higher")
            print("   Or use full dataset (sample_fraction=1.0)")
        else:
            print("❌ Even 10% sample isn't learnable")
            print("\n   POSSIBLE ISSUES:")
            print("   1. Data preprocessing removes useful features")
            print("   2. Class imbalance not properly handled")
            print("   3. Features need different preprocessing")

    if results_large_adam['best_auc'] > 0.65 and results_large_sgd['best_auc'] < 0.55:
        print("\n💡 IMPORTANT: Adam works better than SGD for this task")
        print("   Consider switching optimizer in runner.py")

    if sklearn_large['auc'] > 0.70:
        print(f"\n📊 BASELINE TARGET: sklearn achieves AUC={sklearn_large['auc']:.4f}")
        print("   Neural network should reach at least 90% of this with proper config")

    print(f"\n{'='*80}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
