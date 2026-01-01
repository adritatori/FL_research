"""
Quick Test for Google Colab (T4 GPU optimized)
===============================================
Reduced to 3 clients to work within Colab resource limits
"""

import sys
import os

if __name__ == "__main__":
    print("="*80)
    print("COLAB QUICK TEST - Optimized for T4 GPU")
    print("="*80)
    print("\nThis will run with Colab-friendly parameters:")
    print("  - 1 configuration: ε=5.0, seed=42")
    print("  - 3 rounds (quick)")
    print("  - 2 local epochs (reduced)")
    print("  - 3 clients (fits in T4 VRAM)")
    print("\nEstimated runtime: 3-5 minutes\n")

    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        sys.exit(0)

    # Import and modify the main script
    import run_fairness_validation as main

    # Save originals
    original_rounds = main.NUM_ROUNDS
    original_clients = main.NUM_CLIENTS
    original_epochs = main.LOCAL_EPOCHS
    original_epsilon = main.EPSILON_VALUES
    original_seeds = main.SEEDS

    # Colab-optimized settings
    main.NUM_ROUNDS = 3
    main.NUM_CLIENTS = 3  # Reduced for T4 GPU!
    main.LOCAL_EPOCHS = 2
    main.EPSILON_VALUES = [5.0]
    main.SEEDS = [42]

    try:
        print("\nStarting Colab-optimized test...")
        main.run_all_experiments()
        print("\n" + "="*80)
        print("✓ COLAB TEST PASSED!")
        print("="*80)
        print("\nYour Colab setup is working correctly.")

    except Exception as e:
        print("\n" + "="*80)
        print("✗ COLAB TEST FAILED!")
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
