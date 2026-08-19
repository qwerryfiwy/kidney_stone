"""Statistical significance testing (Section 3.6):
Friedman test followed by pairwise paired t-tests.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats


def run_friedman_test(results: Dict[str, List[float]]) -> Tuple[float, float]:
    """Computes the Friedman test across models.

    Args:
        results: Dict mapping model names to lists of metric values (e.g., AUCs across folds/seeds).
                 Each list must have the same length (number of folds/seeds).
    """
    model_names = list(results.keys())
    if len(model_names) < 3:
        raise ValueError("Friedman test requires at least 3 models to compare.")

    data = [results[name] for name in model_names]

    # Check that all lists have the same length
    lengths = [len(d) for d in data]
    if len(set(lengths)) > 1:
        raise ValueError("All models must have the same number of evaluations (seeds/folds).")

    stat, p_val = stats.friedmanchisquare(*data)
    return stat, p_val


def run_pairwise_ttests(
    results: Dict[str, List[float]]
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Computes pairwise paired t-tests between all pairs of models.

    Args:
        results: Dict mapping model names to lists of metric values.
    """
    model_names = list(results.keys())
    pairwise_results = {}
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            stat, p_val = stats.ttest_rel(results[m1], results[m2])
            pairwise_results[(m1, m2)] = (stat, p_val)
            # Symmetric t-test (negative statistic since direction is inverted)
            pairwise_results[(m2, m1)] = (-stat, p_val)
    return pairwise_results


def print_statistical_report(results: Dict[str, List[float]]) -> None:
    """Computes and prints a formatted statistical significance report."""
    model_names = list(results.keys())
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE REPORT")
    print("=" * 70)

    # Summary stats
    print("\nSummary Statistics:")
    for name in model_names:
        vals = results[name]
        print(
            f"  {name:<20} | Mean: {np.mean(vals):.4f} | Std: {np.std(vals):.4f} "
            f"| Min: {np.min(vals):.4f} | Max: {np.max(vals):.4f}"
        )

    if len(model_names) >= 3:
        try:
            f_stat, f_p = run_friedman_test(results)
            print(f"\nFriedman Test (Comparing all models simultaneously):")
            print(f"  Statistic: {f_stat:.4f}")
            print(
                f"  p-value:   {f_p:.4e} "
                f"({'Significant' if f_p < 0.05 else 'Not Significant'} at alpha=0.05)"
            )
        except Exception as e:
            print(f"\nFriedman Test failed: {e}")
    else:
        print("\nFriedman Test skipped (requires at least 3 models).")

    if len(model_names) >= 2:
        print("\nPairwise Paired t-tests (p-values):")
        pairwise = run_pairwise_ttests(results)

        # Print as a nice matrix/table
        print(f"{'Model':<20}", end="")
        for name in model_names:
            print(f" | {name[:12]:<12}", end="")
        print()
        print("-" * (20 + 15 * len(model_names)))

        for m1 in model_names:
            print(f"{m1:<20}", end="")
            for m2 in model_names:
                if m1 == m2:
                    print(" | -           ", end="")
                else:
                    _, p_val = pairwise[(m1, m2)]
                    print(f" | {p_val:.4e}", end="")
            print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run Friedman and pairwise paired t-tests on model results."
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to JSON file containing {model_name: [metric_values]}",
    )
    args = parser.parse_args()

    with open(args.json, "r") as f:
        results = json.load(f)
    print_statistical_report(results)


if __name__ == "__main__":
    main()
