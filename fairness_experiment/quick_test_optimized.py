"""
Optimized Quick Test - Faster version for testing
=================================================

This version uses smaller batch sizes and fewer clients to speed up testing.
"""

import sys
import os

if __name__ == "__main__":
    print("="*80)
    print("OPTIMIZED QUICK TEST - Fairness Validation Experiment")
    print("="*80)
    print("\nThis will run with optimized parameters for faster execution:")
    print("  - 1 configuration: ε=5.0, seed=42")
    print("  - 3 rounds (instead of 50)")
    print("  - 2 local epochs (instead of 5)")
    print("  - 3 clients (instead of 10)")
    print("  - Batch size: 128 (reduced from 256)")
    print("\nEstimated runtime: 3-5 minutes")
    print("\nNote: DP training is inherently slow due to per-sample gradient computation.")
    print("      Progress logs will show training is advancing.\n")

    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        sys.exit(0)

    # Import and modify the main script
    import run_fairness_validation as main

    # Save original parameters
    original_rounds = main.NUM_ROUNDS
    original_clients = main.NUM_CLIENTS
    original_epochs = main.LOCAL_EPOCHS
    original_epsilon = main.EPSILON_VALUES
    original_seeds = main.SEEDS
    original_batch_size = main.BATCH_SIZE

    # Set optimized parameters
    main.NUM_ROUNDS = 3
    main.NUM_CLIENTS = 3
    main.LOCAL_EPOCHS = 2
    main.EPSILON_VALUES = [5.0]
    main.SEEDS = [42]
    main.BATCH_SIZE = 128  # Smaller batch = faster per-batch computation

    try:
        print("\nStarting optimized quick test...")
        print("Watch for progress logs showing 'Epoch X/Y, Batch Z'...\n")
        main.run_all_experiments()
        print("\n" + "="*80)
        print("✓ OPTIMIZED QUICK TEST PASSED!")
        print("="*80)
        print("\nYour setup is working correctly.")
        print("\nFor the full experiment:")
        print("  - Expect 30-60 minutes per experiment with DP")
        print("  - Total runtime: 6-12 hours for all 12 experiments")
        print("  - Consider running experiments in parallel on multiple machines")
        print("\nTo run full experiment: python run_fairness_validation.py")

    except Exception as e:
        print("\n" + "="*80)
        print("✗ OPTIMIZED QUICK TEST FAILED!")
        print("="*80)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Restore original parameters
        main.NUM_ROUNDS = original_rounds
        main.NUM_CLIENTS = original_clients
        main.LOCAL_EPOCHS = original_epochs
        main.EPSILON_VALUES = original_epsilon
        main.SEEDS = original_seeds
        main.BATCH_SIZE = original_batch_size
