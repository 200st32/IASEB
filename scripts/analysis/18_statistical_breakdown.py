"""
Extended Statistical Analysis for CVPR Rebuttal.

Generates:
1. Referral vs Freeform CIs per model (aggregated across datasets)
2. Spatial vs Temporal CIs per model
3. Entity type breakdown CIs
4. Paired Wilcoxon significance tests
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent / "_archive/2025-01-25/adaw_eval_results_complete_categorized.csv"
OUTPUT_DIR = Path(__file__).parent.parent.parent

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


def format_ci(mean, ci_lower, ci_upper, as_pct=True):
    """Format CI as string."""
    if np.isnan(mean):
        return "N/A"
    if as_pct:
        return f"{mean*100:.2f}% [{ci_lower*100:.2f}, {ci_upper*100:.2f}]"
    return f"{mean:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"


def compute_task_cis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CIs for Referral vs Freeform per model (aggregated across datasets)."""
    results = []

    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]

        for task in ['referral', 'freeform']:
            task_df = model_df[model_df['task'] == task]
            scores = task_df['mvIoU'].dropna().values

            if len(scores) > 0:
                mean, ci_lower, ci_upper, std = bootstrap_ci(scores)
                results.append({
                    'model': model,
                    'task': task,
                    'n_samples': len(scores),
                    'mean_mvIoU': mean,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'ci_width': ci_upper - ci_lower,
                    'std': std
                })

    return pd.DataFrame(results)


def compute_st_cis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CIs for Spatial vs Temporal per model."""
    results = []

    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]

        for st_cls in sorted(df['st_level0_cls'].dropna().unique()):
            st_df = model_df[model_df['st_level0_cls'] == st_cls]
            scores = st_df['mvIoU'].dropna().values

            if len(scores) > 0:
                mean, ci_lower, ci_upper, std = bootstrap_ci(scores)
                results.append({
                    'model': model,
                    'st_class': st_cls,
                    'n_samples': len(scores),
                    'mean_mvIoU': mean,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'ci_width': ci_upper - ci_lower,
                    'std': std
                })

    return pd.DataFrame(results)


def compute_entity_cis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CIs for Entity types per model."""
    results = []

    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]

        for ent_cls in sorted(df['entity_level0_cls'].dropna().unique()):
            ent_df = model_df[model_df['entity_level0_cls'] == ent_cls]
            scores = ent_df['mvIoU'].dropna().values

            if len(scores) > 0:
                mean, ci_lower, ci_upper, std = bootstrap_ci(scores)
                results.append({
                    'model': model,
                    'entity_class': ent_cls,
                    'n_samples': len(scores),
                    'mean_mvIoU': mean,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'ci_width': ci_upper - ci_lower,
                    'std': std
                })

    return pd.DataFrame(results)


def mann_whitney_test(df: pd.DataFrame, group_col: str, val1: str, val2: str) -> dict:
    """
    Perform Mann-Whitney U test (unpaired) comparing two independent groups.
    Appropriate for referral vs freeform since they have different captions/semantics.
    """
    results = {}

    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]

        scores1 = model_df[model_df[group_col] == val1]['mvIoU'].dropna().values
        scores2 = model_df[model_df[group_col] == val2]['mvIoU'].dropna().values

        if len(scores1) > 10 and len(scores2) > 10:
            try:
                stat, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                # Compute effect size (rank-biserial correlation)
                n1, n2 = len(scores1), len(scores2)
                effect_size = 1 - (2 * stat) / (n1 * n2)

                results[model] = {
                    'n_referral': len(scores1),
                    'n_freeform': len(scores2),
                    'mean_referral': np.mean(scores1),
                    'mean_freeform': np.mean(scores2),
                    'mean_diff': np.mean(scores1) - np.mean(scores2),
                    'statistic': stat,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significant_0.05': p_value < 0.05,
                    'significant_0.01': p_value < 0.01
                }
            except ValueError as e:
                results[model] = {'error': str(e)}
        else:
            results[model] = {'error': 'insufficient samples'}

    return results


def model_comparison_wilcoxon(df: pd.DataFrame) -> pd.DataFrame:
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
                except ValueError:
                    pass

    return pd.DataFrame(results)


def main():
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Tasks: {sorted(df['task'].unique())}")
    print(f"ST classes: {sorted(df['st_level0_cls'].dropna().unique())}")
    print(f"Entity classes: {sorted(df['entity_level0_cls'].dropna().unique())}")

    # ========== 1. REFERRAL VS FREEFORM CIs ==========
    print("\n" + "="*60)
    print("1. REFERRAL vs FREEFORM CIs (aggregated per model)")
    print("="*60)

    task_cis = compute_task_cis(df)
    task_cis.to_csv(OUTPUT_DIR / 'referral_vs_freeform_cis.csv', index=False)

    # Pivot for display
    task_pivot = task_cis.pivot(index='model', columns='task', values=['mean_mvIoU', 'ci_lower', 'ci_upper', 'n_samples'])

    print("\nModel             | Freeform                      | Referral")
    print("-"*75)
    for model in sorted(df['model'].unique()):
        model_data = task_cis[task_cis['model'] == model]
        ff = model_data[model_data['task'] == 'freeform']
        ref = model_data[model_data['task'] == 'referral']

        ff_str = format_ci(ff['mean_mvIoU'].values[0], ff['ci_lower'].values[0], ff['ci_upper'].values[0]) if len(ff) > 0 else "N/A"
        ref_str = format_ci(ref['mean_mvIoU'].values[0], ref['ci_lower'].values[0], ref['ci_upper'].values[0]) if len(ref) > 0 else "N/A"

        print(f"{model:17s} | {ff_str:28s} | {ref_str}")

    # ========== 2. SPATIAL VS TEMPORAL CIs ==========
    print("\n" + "="*60)
    print("2. SPATIAL vs TEMPORAL CIs (per model)")
    print("="*60)

    st_cis = compute_st_cis(df)
    st_cis.to_csv(OUTPUT_DIR / 'spatial_vs_temporal_cis.csv', index=False)

    print("\nModel             | Spatial (Static)              | Temporal")
    print("-"*75)
    for model in sorted(df['model'].unique()):
        model_data = st_cis[st_cis['model'] == model]
        spatial = model_data[model_data['st_class'] == 'Spatial (Static)']
        temporal = model_data[model_data['st_class'] == 'Temporal']

        sp_str = format_ci(spatial['mean_mvIoU'].values[0], spatial['ci_lower'].values[0], spatial['ci_upper'].values[0]) if len(spatial) > 0 else "N/A"
        tp_str = format_ci(temporal['mean_mvIoU'].values[0], temporal['ci_lower'].values[0], temporal['ci_upper'].values[0]) if len(temporal) > 0 else "N/A"

        print(f"{model:17s} | {sp_str:28s} | {tp_str}")

    # ========== 3. ENTITY TYPE CIs ==========
    print("\n" + "="*60)
    print("3. ENTITY TYPE CIs (per model)")
    print("="*60)

    entity_cis = compute_entity_cis(df)
    entity_cis.to_csv(OUTPUT_DIR / 'entity_type_cis.csv', index=False)

    # Show abbreviated table
    entity_classes = sorted(df['entity_level0_cls'].dropna().unique())

    print(f"\n{'Model':17s} | " + " | ".join([f"{c[:12]:12s}" for c in entity_classes[:4]]))
    print("-" * 100)
    for model in sorted(df['model'].unique()):
        model_data = entity_cis[entity_cis['model'] == model]
        row = [f"{model:17s}"]
        for ent in entity_classes[:4]:
            ent_data = model_data[model_data['entity_class'] == ent]
            if len(ent_data) > 0:
                row.append(f"{ent_data['mean_mvIoU'].values[0]*100:5.1f}%")
            else:
                row.append("  N/A")
        print(" | ".join(row))

    # ========== 4. MANN-WHITNEY U TESTS: Referral vs Freeform ==========
    print("\n" + "="*60)
    print("4. MANN-WHITNEY U TESTS: Referral vs Freeform (unpaired)")
    print("="*60)

    mw_rf = mann_whitney_test(df, 'task', 'referral', 'freeform')

    print("\nModel             | N(Ref) | N(Free) | Ref Mean | Free Mean | Diff    | p-value  | Sig")
    print("-"*95)
    wilcoxon_results = []
    for model, result in mw_rf.items():
        if 'p_value' in result:
            sig_str = "***" if result['p_value'] < 0.001 else ("**" if result['p_value'] < 0.01 else ("*" if result['p_value'] < 0.05 else ""))
            print(f"{model:17s} | {result['n_referral']:6d} | {result['n_freeform']:7d} | {result['mean_referral']*100:7.2f}% | {result['mean_freeform']*100:8.2f}% | {result['mean_diff']*100:+6.2f}% | {result['p_value']:.2e} | {sig_str}")
            wilcoxon_results.append({
                'model': model,
                'comparison': 'referral_vs_freeform',
                **result
            })
        else:
            print(f"{model:17s} | {result.get('error', 'error')}")

    # ========== 5. MODEL PAIRWISE COMPARISONS ==========
    print("\n" + "="*60)
    print("5. PAIRWISE MODEL COMPARISONS (Wilcoxon)")
    print("="*60)

    model_pairwise = model_comparison_wilcoxon(df)
    model_pairwise.to_csv(OUTPUT_DIR / 'model_pairwise_wilcoxon.csv', index=False)

    # Show significant comparisons only
    significant = model_pairwise[model_pairwise['significant_0.05']]
    print(f"\nSignificant pairwise differences (p < 0.05): {len(significant)} / {len(model_pairwise)}")

    if len(significant) > 0:
        print("\nTop 10 significant comparisons:")
        print("Model 1           | Model 2           | Mean Diff | p-value")
        print("-"*65)
        for _, row in significant.head(10).iterrows():
            print(f"{row['model_1']:17s} | {row['model_2']:17s} | {row['mean_diff']:+8.4f} | {row['p_value']:.2e}")

    # ========== SUMMARY FOR REBUTTAL ==========
    print("\n" + "="*60)
    print("SUMMARY FOR REBUTTAL")
    print("="*60)

    # Overall model ranking with CIs
    print("\nTable 1: Overall Model Performance (95% CI)")
    print("-"*60)
    overall_cis = []
    for model in sorted(df['model'].unique()):
        scores = df[df['model'] == model]['mvIoU'].dropna().values
        mean, ci_lower, ci_upper, std = bootstrap_ci(scores)
        overall_cis.append({
            'model': model,
            'mean': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n': len(scores)
        })

    overall_df = pd.DataFrame(overall_cis).sort_values('mean', ascending=False)
    for _, row in overall_df.iterrows():
        print(f"{row['model']:17s}: {row['mean']*100:5.2f}% [{row['ci_lower']*100:.2f}, {row['ci_upper']*100:.2f}] (n={row['n']})")

    overall_df.to_csv(OUTPUT_DIR / 'overall_model_cis.csv', index=False)

    # Save Wilcoxon results
    pd.DataFrame(wilcoxon_results).to_csv(OUTPUT_DIR / 'wilcoxon_referral_freeform.csv', index=False)

    print("\n" + "="*60)
    print("FILES SAVED:")
    print("="*60)
    print(f"  - {OUTPUT_DIR / 'referral_vs_freeform_cis.csv'}")
    print(f"  - {OUTPUT_DIR / 'spatial_vs_temporal_cis.csv'}")
    print(f"  - {OUTPUT_DIR / 'entity_type_cis.csv'}")
    print(f"  - {OUTPUT_DIR / 'model_pairwise_wilcoxon.csv'}")
    print(f"  - {OUTPUT_DIR / 'wilcoxon_referral_freeform.csv'}")
    print(f"  - {OUTPUT_DIR / 'overall_model_cis.csv'}")


if __name__ == "__main__":
    main()
