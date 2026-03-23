"""
Bootstrap 95% Confidence Interval Analysis for mvIoU scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent / "_archive/2025-01-25/adaw_eval_results_complete_categorized.csv"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "bootstrap_ci_results.csv"

N_BOOTSTRAP = 10000
CI_LEVEL = 0.95
RANDOM_SEED = 42


def bootstrap_ci(scores: np.ndarray, n_bootstrap: int = N_BOOTSTRAP, ci_level: float = CI_LEVEL) -> tuple:
    """
    Compute bootstrap confidence interval for the mean.

    Returns: (mean, ci_lower, ci_upper, std)
    """
    np.random.seed(RANDOM_SEED)
    n = len(scores)

    if n == 0:
        return np.nan, np.nan, np.nan, np.nan

    if n == 1:
        return scores[0], scores[0], scores[0], 0.0

    # Bootstrap resampling
    bootstrap_means = np.array([
        np.mean(np.random.choice(scores, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    # Percentile method for CI
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return np.mean(scores), ci_lower, ci_upper, np.std(scores)


def main():
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples")

    # Show unique values for grouping columns
    print(f"\nModels: {df['model'].nunique()} unique")
    print(f"Datasets: {df['dataset'].nunique()} unique")
    print(f"Tasks: {df['task'].nunique()} unique")
    print(f"ST classes: {df['st_level0_cls'].nunique()} unique")
    print(f"Entity classes: {df['entity_level0_cls'].nunique()} unique")

    results = []

    # Group by model, dataset, task
    print("\n--- Computing Bootstrap 95% CIs ---")
    groupby_cols = ['model', 'dataset', 'task']

    grouped = df.groupby(groupby_cols)
    total_groups = len(grouped)

    for i, (group_key, group_df) in enumerate(grouped):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Processing group {i+1}/{total_groups}")

        scores = group_df['mvIoU'].dropna().values
        mean, ci_lower, ci_upper, std = bootstrap_ci(scores)

        result = {
            'model': group_key[0],
            'dataset': group_key[1],
            'task': group_key[2],
            'n_samples': len(scores),
            'mean_mvIoU': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'std': std,
        }
        results.append(result)

    # Also compute by ST class
    print("\n--- Computing CIs by ST class ---")
    for (model, st_cls), group_df in df.groupby(['model', 'st_level0_cls']):
        scores = group_df['mvIoU'].dropna().values
        mean, ci_lower, ci_upper, std = bootstrap_ci(scores)

        result = {
            'model': model,
            'dataset': 'ALL',
            'task': f'st:{st_cls}',
            'n_samples': len(scores),
            'mean_mvIoU': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'std': std,
        }
        results.append(result)

    # Also compute by Entity class
    print("\n--- Computing CIs by Entity class ---")
    for (model, ent_cls), group_df in df.groupby(['model', 'entity_level0_cls']):
        scores = group_df['mvIoU'].dropna().values
        mean, ci_lower, ci_upper, std = bootstrap_ci(scores)

        result = {
            'model': model,
            'dataset': 'ALL',
            'task': f'entity:{ent_cls}',
            'n_samples': len(scores),
            'mean_mvIoU': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'std': std,
        }
        results.append(result)

    # Overall by model
    print("\n--- Computing overall CIs by model ---")
    for model, group_df in df.groupby('model'):
        scores = group_df['mvIoU'].dropna().values
        mean, ci_lower, ci_upper, std = bootstrap_ci(scores)

        result = {
            'model': model,
            'dataset': 'ALL',
            'task': 'ALL',
            'n_samples': len(scores),
            'mean_mvIoU': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'std': std,
        }
        results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['model', 'dataset', 'task'])
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(results_df)} rows to {OUTPUT_PATH}")

    # Print summary
    print("\n=== OVERALL MODEL RESULTS (95% CI) ===")
    overall = results_df[(results_df['dataset'] == 'ALL') & (results_df['task'] == 'ALL')]
    overall = overall.sort_values('mean_mvIoU', ascending=False)
    for _, row in overall.iterrows():
        print(f"{row['model']:20s}: {row['mean_mvIoU']:.4f} [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] (n={row['n_samples']})")


if __name__ == "__main__":
    main()
