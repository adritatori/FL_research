#!/usr/bin/env python3
"""
Sequential (non-parallel) test runner for debugging
Avoids Ray OOM issues by running experiments one at a time
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main script components
from test_config import TestConfig

# Modify the main script to use test config and run sequentially
def run_test():
    """Run a minimal test without Ray parallelism"""

    print("="*80)
    print("SEQUENTIAL TEST MODE (No Ray)")
    print("Safer for memory-constrained environments")
    print("="*80)

    # Initialize config
    config = TestConfig()
    config.setup_directories()
    config.set_seed()
    config.print_summary()

    print("\n⚠️  To use this configuration:")
    print("1. Edit nwdaf_security_analytics_.py")
    print("2. Change the configuration at the top to use TestConfig:")
    print("   from test_config import TestConfig")
    print("   config = TestConfig()")
    print("3. Set AGGREGATORS = ['fedavg', 'trimmed_mean']")
    print("4. Limit Ray CPUs: ray.init(num_cpus=2, ...)")
    print()
    print("OR use the quick fix below:")
    print("-"*80)

if __name__ == "__main__":
    run_test()
