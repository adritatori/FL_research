"""
Diagnostic Test - Check if training works without DP
"""

import sys
import os

if __name__ == "__main__":
    print("="*80)
    print("DIAGNOSTIC TEST - No DP Training")
    print("="*80)
    print("\nThis will test if the basic training works without differential privacy.")
    print("If this works, the issue is likely with Opacus/DP overhead.")
    print("\nConfiguration:")
    print("  - No DP (ε=∞)")
    print("  - 2 rounds")
    print("  - 2 local epochs")
    print("  - 3 clients")
    print("  - Batch size: 128 (reduced)")
    print("\nEstimated runtime: 2-3 minutes\n")

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

    # Set diagnostic parameters
    main.NUM_ROUNDS = 2
    main.NUM_CLIENTS = 3
    main.LOCAL_EPOCHS = 2
    main.EPSILON_VALUES = [float('inf')]  # No DP
    main.SEEDS = [42]
    main.BATCH_SIZE = 128  # Smaller batch size

    try:
        print("\nStarting diagnostic test...\n")
        main.run_all_experiments()
        print("\n" + "="*80)
        print("✓ DIAGNOSTIC TEST PASSED!")
        print("="*80)
        print("\nBasic training works. The issue with quick_test.py is likely:")
        print("  1. Opacus/DP overhead is very slow with 194 features")
        print("  2. Consider reducing batch size or number of clients")
        print("  3. Or be patient - DP training can take 10-20x longer")

    except Exception as e:
        print("\n" + "="*80)
        print("✗ DIAGNOSTIC TEST FAILED!")
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
