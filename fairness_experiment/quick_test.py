"""
Quick Test Script for Fairness Validation Experiment
====================================================

This script runs a single quick experiment to verify the setup is working.
It uses reduced parameters for faster execution (~5-10 minutes).

To test the full setup, run:
    python quick_test.py

To run the full experiment suite, run:
    python run_fairness_validation.py
"""

import sys
import os

# Modify the main script's constants for quick testing
if __name__ == "__main__":
    print("="*80)
    print("QUICK TEST - Fairness Validation Experiment")
    print("="*80)
    print("\nThis will run a single experiment with reduced parameters:")
    print("  - 1 configuration: ε=5.0, seed=42")
    print("  - 5 rounds (instead of 50)")
    print("  - 3 local epochs (instead of 5)")
    print("  - 5 clients (instead of 10)")
    print("\nEstimated runtime: 5-10 minutes\n")

    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        sys.exit(0)

    # Import and modify the main script
    import run_fairness_validation as main

    # Temporarily modify parameters
    original_rounds = main.NUM_ROUNDS
    original_clients = main.NUM_CLIENTS
    original_epochs = main.LOCAL_EPOCHS
    original_epsilon = main.EPSILON_VALUES
    original_seeds = main.SEEDS

    main.NUM_ROUNDS = 5
    main.NUM_CLIENTS = 5
    main.LOCAL_EPOCHS = 3
    main.EPSILON_VALUES = [5.0]
    main.SEEDS = [42]

    try:
        print("\nStarting quick test...")
        main.run_all_experiments()
        print("\n" + "="*80)
        print("✓ QUICK TEST PASSED!")
        print("="*80)
        print("\nYour setup is working correctly.")
        print("To run the full experiment suite, execute:")
        print("    python run_fairness_validation.py")

    except Exception as e:
        print("\n" + "="*80)
        print("✗ QUICK TEST FAILED!")
        print("="*80)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease check:")
        print("  1. Data files are present (UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv)")
        print("  2. All dependencies are installed (pip install -r requirements.txt)")
        print("  3. Python version >= 3.8")

    finally:
        # Restore original parameters
        main.NUM_ROUNDS = original_rounds
        main.NUM_CLIENTS = original_clients
        main.LOCAL_EPOCHS = original_epochs
        main.EPSILON_VALUES = original_epsilon
        main.SEEDS = original_seeds
