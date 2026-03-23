#!/usr/bin/env python3
"""
Statistical Analysis for Rebuttal - Generates key numbers.

Uses the combined results CSV with Qwen3-VL.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent / "results/combined_results_with_qwen3vl.csv"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "results"

N_BOOTSTRAP = 10000
CI_LEVEL = 0.95
RANDOM_SEED = 42


def bootstrap_ci(scores: np.ndarray, n_bootstrap: int = N_BOOTSTRAP, ci_level: float = CI_LEVEL) -> tuple:
    """Compute bootstrap confidence interval for the mean."""
    np.random.seed(RANDOM_SEED)
    n = len(scores)

    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    if n == 1:
        return scores[0], scores[0], scores[0], 0.0

    bootstrap_means = np.array([
        np.mean(np.random.choice(scores, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return np.mean(scores), ci_lower, ci_upper, np.std(scores)


def model_pairwise_wilcoxon(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Wilcoxon tests between models on same samples."""
    results = []
    models = sorted(df['model'].unique())

    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            # Get overlapping samples
            m1_df = df[df['model'] == model1][['video_path', 'caption', 'dataset', 'task', 'mvIoU']].copy()
            m2_df = df[df['model'] == model2][['video_path', 'caption', 'dataset', 'task', 'mvIoU']].copy()

            m1_df = m1_df.rename(columns={'mvIoU': 'mvIoU_1'})
            m2_df = m2_df.rename(columns={'mvIoU': 'mvIoU_2'})

            merged = pd.merge(m1_df, m2_df, on=['video_path', 'caption', 'dataset', 'task'], how='inner')

            # Drop pairs where either has NaN
            merged = merged.dropna(subset=['mvIoU_1', 'mvIoU_2'])

            if len(merged) > 100:
                scores1 = merged['mvIoU_1'].values
                scores2 = merged['mvIoU_2'].values

                try:
                    stat, p_value = stats.wilcoxon(scores1, scores2, alternative='two-sided')
                    results.append({
                        'model_1': model1,
                        'model_2': model2,
                        'n_pairs': len(merged),
                        'mean_1': np.mean(scores1),
                        'mean_2': np.mean(scores2),
                        'mean_diff': np.mean(scores1 - scores2),
                        'statistic': stat,
                        'p_value': p_value,
                        'significant_0.05': p_value < 0.05,
                        'significant_0.01': p_value < 0.01
                    })
                except ValueError as e:
                    results.append({
                        'model_1': model1,
                        'model_2': model2,
                        'n_pairs': len(merged),
                        'error': str(e)
                    })

    return pd.DataFrame(results)


def mann_whitney_referral_freeform(df: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U test comparing Referral vs Freeform per model."""
    results = []

    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]

        ref_scores = model_df[model_df['task'] == 'referral']['mvIoU'].dropna().values
        ff_scores = model_df[model_df['task'] == 'freeform']['mvIoU'].dropna().values

        if len(ref_scores) > 10 and len(ff_scores) > 10:
            try:
                stat, p_value = stats.mannwhitneyu(ref_scores, ff_scores, alternative='two-sided')
                better_on = 'referral' if np.mean(ref_scores) > np.mean(ff_scores) else 'freeform'

                results.append({
                    'model': model,
                    'n_referral': len(ref_scores),
                    'n_freeform': len(ff_scores),
                    'mean_referral': np.mean(ref_scores),
                    'mean_freeform': np.mean(ff_scores),
                    'mean_diff': np.mean(ref_scores) - np.mean(ff_scores),
                    'better_on': better_on,
                    'statistic': stat,
                    'p_value': p_value,
                    'significant_0.05': p_value < 0.05
                })
            except ValueError as e:
                results.append({'model': model, 'error': str(e)})
        else:
            results.append({
                'model': model,
                'n_referral': len(ref_scores),
                'n_freeform': len(ff_scores),
                'error': 'insufficient_samples'
            })

    return pd.DataFrame(results)


def main():
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")

    # ========== 1. OVERALL MODEL RANKING WITH CIs ==========
    print("\n" + "="*70)
    print("1. OVERALL MODEL RANKING (95% Bootstrap CI)")
    print("="*70)

    overall_results = []
    for model in sorted(df['model'].unique()):
        scores = df[df['model'] == model]['mvIoU'].dropna().values
        mean, ci_lower, ci_upper, std = bootstrap_ci(scores)
        overall_results.append({
            'model': model,
            'n_samples': len(scores),
            'mean_mvIoU': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': std
        })

    overall_df = pd.DataFrame(overall_results).sort_values('mean_mvIoU', ascending=False)
    overall_df['rank'] = range(1, len(overall_df) + 1)
    overall_df.to_csv(OUTPUT_DIR / 'overall_model_ranking.csv', index=False)

    print("\nRank | Model           | Mean mvIoU | 95% CI                | n")
    print("-"*70)
    for _, row in overall_df.iterrows():
        print(f"{row['rank']:4d} | {row['model']:15s} | {row['mean_mvIoU']*100:6.2f}%   | [{row['ci_lower']*100:.2f}%, {row['ci_upper']*100:.2f}%] | {row['n_samples']}")

    # ========== 2. PAIRWISE WILCOXON ==========
    print("\n" + "="*70)
    print("2. PAIRWISE MODEL COMPARISONS (Wilcoxon Signed-Rank)")
    print("="*70)

    pairwise_df = model_pairwise_wilcoxon(df)
    pairwise_df.to_csv(OUTPUT_DIR / 'model_pairwise_wilcoxon.csv', index=False)

    n_comparisons = len(pairwise_df)
    n_significant = len(pairwise_df[pairwise_df.get('significant_0.05', False) == True])

    print(f"\nSignificant pairwise differences: {n_significant} / {n_comparisons}")

    # Show Qwen3-VL comparisons
    qwen3vl_pairs = pairwise_df[(pairwise_df['model_1'] == 'qwen3vl') | (pairwise_df['model_2'] == 'qwen3vl')]
    if len(qwen3vl_pairs) > 0:
        print("\nQwen3-VL vs other models:")
        print("Comparison               | Mean Diff | p-value    | Sig")
        print("-"*60)
        for _, row in qwen3vl_pairs.iterrows():
            other = row['model_2'] if row['model_1'] == 'qwen3vl' else row['model_1']
            diff = row.get('mean_diff', np.nan)
            if row['model_1'] != 'qwen3vl':
                diff = -diff  # Flip sign if qwen3vl is model_2
            p = row.get('p_value', np.nan)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"qwen3vl vs {other:12s} | {diff*100:+6.2f}%   | {p:.2e} | {sig}")

    # ========== 3. REFERRAL VS FREEFORM ==========
    print("\n" + "="*70)
    print("3. REFERRAL vs FREEFORM (Mann-Whitney U)")
    print("="*70)

    rf_df = mann_whitney_referral_freeform(df)
    rf_df.to_csv(OUTPUT_DIR / 'referral_vs_freeform_mannwhitney.csv', index=False)

    valid_tests = rf_df[rf_df['p_value'].notna()]
    n_significant_rf = len(valid_tests[valid_tests['significant_0.05'] == True])
    n_tests = len(valid_tests)

    print(f"\nSignificant R vs F differences: {n_significant_rf} / {n_tests}")

    print("\nModel           | Better On | R Mean  | F Mean  | Diff    | p-value    | Sig")
    print("-"*80)
    for _, row in rf_df.iterrows():
        if pd.notna(row.get('p_value')):
            p = row['p_value']
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "NS"))
            print(f"{row['model']:15s} | {row['better_on']:9s} | {row['mean_referral']*100:5.2f}% | {row['mean_freeform']*100:5.2f}% | {row['mean_diff']*100:+5.2f}% | {p:.2e} | {sig}")
        else:
            print(f"{row['model']:15s} | {row.get('error', 'N/A')}")

    # ========== 4. DATASET BREAKDOWN ==========
    print("\n" + "="*70)
    print("4. QWEN3-VL PERFORMANCE BY DATASET")
    print("="*70)

    qwen3vl_df = df[df['model'] == 'qwen3vl']
    print("\nDataset   | n      | Mean mvIoU | 95% CI")
    print("-"*50)
    for dataset in sorted(qwen3vl_df['dataset'].unique()):
        ds_scores = qwen3vl_df[qwen3vl_df['dataset'] == dataset]['mvIoU'].dropna().values
        mean, ci_lower, ci_upper, _ = bootstrap_ci(ds_scores)
        print(f"{dataset:9s} | {len(ds_scores):6d} | {mean*100:6.2f}%    | [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

    # ========== REBUTTAL SUMMARY ==========
    print("\n" + "="*70)
    print("REBUTTAL SUMMARY")
    print("="*70)

    summary_lines = []

    # Key numbers
    qwen3vl_rank = overall_df[overall_df['model'] == 'qwen3vl']['rank'].values[0]
    qwen3vl_mean = overall_df[overall_df['model'] == 'qwen3vl']['mean_mvIoU'].values[0]
    qwen3vl_ci_l = overall_df[overall_df['model'] == 'qwen3vl']['ci_lower'].values[0]
    qwen3vl_ci_u = overall_df[overall_df['model'] == 'qwen3vl']['ci_upper'].values[0]

    top_model = overall_df.iloc[0]['model']
    top_mean = overall_df.iloc[0]['mean_mvIoU']

    summary_lines.append(f"1. Qwen3-VL ranks #{qwen3vl_rank} with {qwen3vl_mean*100:.1f}% mean mvIoU (95% CI: [{qwen3vl_ci_l*100:.1f}%, {qwen3vl_ci_u*100:.1f}%])")
    summary_lines.append(f"2. Top model: {top_model} at {top_mean*100:.1f}% mean mvIoU")
    summary_lines.append(f"3. Pairwise significance: {n_significant}/{n_comparisons} model pairs show significant differences (p<0.05)")
    summary_lines.append(f"4. Referral vs Freeform: {n_significant_rf}/{n_tests} models show significant task-type differences")

    # Models better on referral vs freeform
    better_referral = valid_tests[(valid_tests['significant_0.05'] == True) & (valid_tests['better_on'] == 'referral')]
    better_freeform = valid_tests[(valid_tests['significant_0.05'] == True) & (valid_tests['better_on'] == 'freeform')]

    summary_lines.append(f"   - {len(better_referral)} models significantly better on Referral")
    summary_lines.append(f"   - {len(better_freeform)} models significantly better on Freeform")

    for line in summary_lines:
        print(line)

    # Save summary to file
    with open(OUTPUT_DIR / 'rebuttal_summary.txt', 'w') as f:
        f.write("IASEB Rebuttal Statistics Summary\n")
        f.write(f"Generated from: {DATA_PATH}\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Models: {len(df['model'].unique())}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*70 + "\n\n")
        for line in summary_lines:
            f.write(line + "\n")

        f.write("\n" + "="*70 + "\n")
        f.write("FULL MODEL RANKING\n")
        f.write("="*70 + "\n\n")
        for _, row in overall_df.iterrows():
            f.write(f"{row['rank']}. {row['model']}: {row['mean_mvIoU']*100:.2f}% [{row['ci_lower']*100:.2f}%, {row['ci_upper']*100:.2f}%] (n={row['n_samples']})\n")

    print(f"\nSaved: {OUTPUT_DIR / 'rebuttal_summary.txt'}")
    print(f"Saved: {OUTPUT_DIR / 'overall_model_ranking.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'model_pairwise_wilcoxon.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'referral_vs_freeform_mannwhitney.csv'}")


if __name__ == "__main__":
    main()
