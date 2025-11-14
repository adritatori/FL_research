"""
Run Analysis Only
=================
Re-run statistical analysis on existing experiment results.

Use this when:
- Experiments are done but analysis failed
- You want to re-analyze with different metrics
- You want to regenerate plots/tables

Usage:
    python run_analysis_only.py
"""

import sys
from pathlib import Path

# Import modules
from config import config
from analysis import analyze_phase_results, create_summary_table, compare_aggregators


def main():
    """Run analysis only (no experiments)"""

    print("\n" + "="*80)
    print("RE-RUNNING ANALYSIS ONLY")
    print("="*80)

    # Get results directory for current phase
    results_dir = config.get_result_dir()

    print(f"\nPhase: {config.PHASE}")
    print(f"Results directory: {results_dir}")

    # Check if results exist
    result_files = list(results_dir.glob("*.json"))
    result_files = [f for f in result_files if not f.name.startswith('summary') and not f.name.startswith('analysis')]

    if not result_files:
        print(f"\n❌ ERROR: No experiment results found in {results_dir}")
        print(f"   Make sure you've run experiments first with: python main.py")
        sys.exit(1)

    print(f"\n✓ Found {len(result_files)} experiment results")
    print(f"\nRunning statistical analysis...\n")

    # Run analysis for different metrics
    try:
        print("="*80)
        print("ANALYZING: final_f1")
        print("="*80)
        analyze_phase_results(results_dir, 'final_f1')

        print("\n" + "="*80)
        print("ANALYZING: final_accuracy")
        print("="*80)
        analyze_phase_results(results_dir, 'final_accuracy')

        print("\n" + "="*80)
        print("ANALYZING: best_f1")
        print("="*80)
        analyze_phase_results(results_dir, 'best_f1')

    except Exception as e:
        print(f"\n⚠️  Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

    # Create summary table
    try:
        print("\n" + "="*80)
        print("CREATING SUMMARY TABLE")
        print("="*80)
        create_summary_table(results_dir)

    except Exception as e:
        print(f"\n⚠️  Summary table failed: {str(e)}")
        import traceback
        traceback.print_exc()

    # Compare aggregators (if applicable)
    try:
        phase_cfg = config.get_phase_config()
        if len(phase_cfg['configs']) > 1:
            print("\n" + "="*80)
            print("COMPARING AGGREGATORS")
            print("="*80)
            compare_aggregators(results_dir)

    except Exception as e:
        print(f"\n⚠️  Aggregator comparison failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {results_dir / 'analysis'}")
    print(f"\nCheck these files:")
    print(f"  - statistical_analysis_final_f1.json")
    print(f"  - statistical_analysis_final_accuracy.json")
    print(f"  - statistical_analysis_best_f1.json")
    print(f"  - summary_table.csv")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
