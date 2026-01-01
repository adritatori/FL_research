"""Minimal test - CPU only, 2 clients, 2 rounds"""
import sys
sys.path.insert(0, '/home/user/FL_research/fairness_experiment')

import run_fairness_validation as main

# Override to minimal settings
main.NUM_ROUNDS = 2
main.NUM_CLIENTS = 2  # Only 2 clients!
main.LOCAL_EPOCHS = 1
main.EPSILON_VALUES = [float('inf')]  # No DP
main.SEEDS = [42]

print("\n" + "="*80)
print("MINIMAL TEST: 2 clients, 2 rounds, CPU only")
print("="*80)

try:
    main.run_all_experiments()
    print("\n✓ SUCCESS!")
except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
