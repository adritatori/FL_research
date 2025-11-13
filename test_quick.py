#!/usr/bin/env python3
"""
Quick validation test - runs in <5 minutes
Verifies your setup works before running full experiments
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("="*80)
    print("TEST 1: Checking imports...")
    print("="*80)

    try:
        import torch
        print("✓ PyTorch")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False

    try:
        import numpy as np
        print("✓ NumPy")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False

    try:
        import pandas as pd
        print("✓ Pandas")
    except ImportError as e:
        print(f"❌ Pandas: {e}")
        return False

    try:
        import flwr as fl
        print("✓ Flower (FL framework)")
    except ImportError as e:
        print(f"❌ Flower: {e}")
        print("   Install with: pip install flwr[simulation]")
        return False

    try:
        import ray
        print("✓ Ray")
    except ImportError as e:
        print(f"❌ Ray: {e}")
        return False

    try:
        from opacus import PrivacyEngine
        print("✓ Opacus (Differential Privacy)")
    except ImportError as e:
        print(f"❌ Opacus: {e}")
        return False

    try:
        import psutil
        print("✓ psutil")
    except ImportError as e:
        print(f"❌ psutil: {e}")
        return False

    print("\n✅ All imports successful!\n")
    return True


def test_aggregators():
    """Test that aggregation strategies exist"""
    print("="*80)
    print("TEST 2: Checking aggregation strategies...")
    print("="*80)

    try:
        from nwdaf_security_analytics_ import TrimmedMeanStrategy
        print("✓ TrimmedMeanStrategy exists")
    except ImportError as e:
        print(f"❌ TrimmedMeanStrategy: {e}")
        return False

    try:
        from nwdaf_security_analytics_ import MedianStrategy
        print("✓ MedianStrategy exists")
    except ImportError as e:
        print(f"❌ MedianStrategy: {e}")
        return False

    # Krum should NOT exist
    try:
        from nwdaf_security_analytics_ import KrumStrategy
        print("❌ KrumStrategy exists (should have been removed!)")
        return False
    except (ImportError, AttributeError):
        print("✓ KrumStrategy correctly removed")

    print("\n✅ All aggregators correct!\n")
    return True


def test_config():
    """Test configuration"""
    print("="*80)
    print("TEST 3: Checking configuration...")
    print("="*80)

    try:
        from working_test_config import WorkingTestConfig
        config = WorkingTestConfig()

        print(f"✓ Config loaded")
        print(f"  - Clients: {config.NUM_CLIENTS}")
        print(f"  - Rounds: {config.NUM_ROUNDS}")
        print(f"  - Aggregators: {config.AGGREGATORS}")

        # Check for krum
        if 'krum' in config.AGGREGATORS:
            print("❌ Config contains 'krum' aggregator (doesn't exist!)")
            return False

        print("✓ No 'krum' in aggregators")

        # Check valid aggregators
        valid = ['fedavg', 'trimmed_mean', 'median']
        for agg in config.AGGREGATORS:
            if agg not in valid:
                print(f"❌ Unknown aggregator: {agg}")
                return False

        print(f"✓ All aggregators valid: {config.AGGREGATORS}")

    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

    print("\n✅ Configuration valid!\n")
    return True


def test_memory():
    """Check available memory"""
    print("="*80)
    print("TEST 4: Checking system resources...")
    print("="*80)

    try:
        import psutil

        mem = psutil.virtual_memory()
        mem_gb = mem.total / (1024**3)
        mem_avail_gb = mem.available / (1024**3)

        print(f"Total memory: {mem_gb:.1f} GB")
        print(f"Available: {mem_avail_gb:.1f} GB")
        print(f"Used: {mem.percent}%")

        if mem_avail_gb < 4:
            print("⚠️  Warning: Less than 4GB available")
            print("   Recommendation: Close other applications")
            print("   Or use: ray.init(num_cpus=1, ...)")
        else:
            print("✓ Sufficient memory available")

        cpu_count = os.cpu_count()
        print(f"\nCPU cores: {cpu_count}")
        print(f"✓ Recommend: ray.init(num_cpus=2, ...) for your system")

    except Exception as e:
        print(f"⚠️  Could not check memory: {e}")

    print("\n✅ System resources OK!\n")
    return True


def test_ray():
    """Test Ray initialization"""
    print("="*80)
    print("TEST 5: Testing Ray (parallel execution)...")
    print("="*80)

    try:
        import ray

        # Shutdown if already running
        if ray.is_initialized():
            ray.shutdown()

        # Initialize with limited CPUs
        ray.init(num_cpus=2, log_to_driver=False, ignore_reinit_error=True)
        print("✓ Ray initialized successfully (2 CPUs)")

        # Simple test
        @ray.remote
        def test_func(x):
            return x * 2

        result = ray.get(test_func.remote(5))
        if result == 10:
            print("✓ Ray test function works")
        else:
            print("❌ Ray test failed")
            return False

        ray.shutdown()
        print("✓ Ray shutdown successful")

    except Exception as e:
        print(f"❌ Ray error: {e}")
        return False

    print("\n✅ Ray working correctly!\n")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("QUICK VALIDATION TEST")
    print("Verifying setup before running experiments")
    print("="*80 + "\n")

    tests = [
        ("Imports", test_imports),
        ("Aggregators", test_aggregators),
        ("Configuration", test_config),
        ("Memory", test_memory),
        ("Ray", test_ray),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}\n")
            results.append((name, False))

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! You're ready to run experiments.")
        print("\nNext steps:")
        print("1. Ensure AGGREGATORS = ['fedavg', 'trimmed_mean'] (NO 'krum'!)")
        print("2. Use: ray.init(num_cpus=2, ...)")
        print("3. Run your experiment")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("See FIX_RAY_OOM.md for detailed troubleshooting.")

    print("="*80)


if __name__ == "__main__":
    main()
