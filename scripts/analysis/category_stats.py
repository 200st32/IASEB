#!/usr/bin/env python3
"""
Category-wise Statistical Analysis for Rebuttal.

Merges Qwen3-VL results with existing category annotations and generates
breakdown by Entity Type and Spatial/Temporal classification.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
QWEN3VL_CSV = BASE_DIR / "results/combined_results_with_qwen3vl.csv"
CATEGORIZED_CSV = BASE_DIR / "_archive/adaw_eval_results_complete_categorized.csv"
OUTPUT_DIR = BASE_DIR / "results"

N_BOOTSTRAP = 10000
CI_LEVEL = 0.95
RANDOM_SEED = 42


def bootstrap_ci(scores: np.ndarray) -> tuple:
    """Compute bootstrap confidence interval for the mean."""
    np.random.seed(RANDOM_SEED)
    n = len(scores)

    if n == 0:
        return np.nan, np.nan, np.nan
    if n < 5:
        return np.mean(scores), np.nan, np.nan

    bootstrap_means = np.array([
        np.mean(np.random.choice(scores, size=n, replace=True))
        for _ in range(N_BOOTSTRAP)
    ])

    alpha = 1 - CI_LEVEL
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return np.mean(scores), ci_lower, ci_upper


def normalize_caption(caption):
    """Normalize caption for matching."""
    if pd.isna(caption):
        return ""
    return str(caption).strip().lower().replace('.', '').replace('"', '').replace('\\', '')


def main():
    print("="*70)
    print("Category-wise Statistical Analysis")
    print("="*70)

    # Load data
    print(f"\nLoading Qwen3-VL combined CSV: {QWEN3VL_CSV}")
    qwen3vl_df = pd.read_csv(QWEN3VL_CSV)
    print(f"  Total rows: {len(qwen3vl_df)}")
    print(f"  Models: {sorted(qwen3vl_df['model'].unique())}")

    print(f"\nLoading categorized CSV: {CATEGORIZED_CSV}")
    cat_df = pd.read_csv(CATEGORIZED_CSV)
    print(f"  Total rows: {len(cat_df)}")

    # Create category lookup from categorized data
    # Use first occurrence of each (caption, dataset) to get category
    cat_df['caption_norm'] = cat_df['caption'].apply(normalize_caption)

    category_lookup = cat_df.groupby(['caption_norm', 'dataset']).first()[
        ['st_level0_cls', 'entity_level0_cls']
    ].reset_index()

    print(f"\nCategory lookup: {len(category_lookup)} unique (caption, dataset) pairs")
    print(f"  With ST class: {category_lookup['st_level0_cls'].notna().sum()}")
    print(f"  With Entity class: {category_lookup['entity_level0_cls'].notna().sum()}")

    # Normalize captions in qwen3vl data for matching
    qwen3vl_df['caption_norm'] = qwen3vl_df['caption'].apply(normalize_caption)

    # Merge categories into full dataset
    merged_df = qwen3vl_df.merge(
        category_lookup,
        on=['caption_norm', 'dataset'],
        how='left'
    )

    print(f"\nMerged dataset: {len(merged_df)} rows")

    # Check coverage
    qwen3vl_only = merged_df[merged_df['model'] == 'qwen3vl']
    print(f"\nQwen3-VL category coverage:")
    print(f"  Total samples: {len(qwen3vl_only)}")
    print(f"  With ST class: {qwen3vl_only['st_level0_cls'].notna().sum()} ({qwen3vl_only['st_level0_cls'].notna().mean()*100:.1f}%)")
    print(f"  With Entity class: {qwen3vl_only['entity_level0_cls'].notna().sum()} ({qwen3vl_only['entity_level0_cls'].notna().mean()*100:.1f}%)")

    # ========== ENTITY TYPE BREAKDOWN ==========
    print("\n" + "="*70)
    print("ENTITY TYPE BREAKDOWN")
    print("="*70)

    entity_results = []
    entity_classes = sorted(merged_df['entity_level0_cls'].dropna().unique())

    for model in sorted(merged_df['model'].unique()):
        model_df = merged_df[merged_df['model'] == model]

        for entity_cls in entity_classes:
            subset = model_df[model_df['entity_level0_cls'] == entity_cls]
            scores = subset['mvIoU'].dropna().values

            if len(scores) > 0:
                mean, ci_l, ci_u = bootstrap_ci(scores)
                entity_results.append({
                    'model': model,
                    'entity_class': entity_cls,
                    'n_samples': len(scores),
                    'mean_mvIoU': mean,
                    'ci_lower': ci_l,
                    'ci_upper': ci_u
                })

    entity_df = pd.DataFrame(entity_results)
    entity_df.to_csv(OUTPUT_DIR / 'entity_type_breakdown.csv', index=False)

    # Print entity table
    print(f"\n{'Model':<15} | " + " | ".join([f"{c[:12]:>12}" for c in entity_classes]))
    print("-" * (15 + 3 + len(entity_classes) * 15))

    for model in sorted(merged_df['model'].unique()):
        row = [f"{model:<15}"]
        for entity_cls in entity_classes:
            subset = entity_df[(entity_df['model'] == model) & (entity_df['entity_class'] == entity_cls)]
            if len(subset) > 0 and not pd.isna(subset['mean_mvIoU'].values[0]):
                row.append(f"{subset['mean_mvIoU'].values[0]*100:>11.1f}%")
            else:
                row.append(f"{'N/A':>12}")
        print(" | ".join(row))

    # ========== SPATIAL VS TEMPORAL BREAKDOWN ==========
    print("\n" + "="*70)
    print("SPATIAL vs TEMPORAL BREAKDOWN")
    print("="*70)

    st_results = []
    st_classes = sorted(merged_df['st_level0_cls'].dropna().unique())

    for model in sorted(merged_df['model'].unique()):
        model_df = merged_df[merged_df['model'] == model]

        for st_cls in st_classes:
            subset = model_df[model_df['st_level0_cls'] == st_cls]
            scores = subset['mvIoU'].dropna().values

            if len(scores) > 0:
                mean, ci_l, ci_u = bootstrap_ci(scores)
                st_results.append({
                    'model': model,
                    'st_class': st_cls,
                    'n_samples': len(scores),
                    'mean_mvIoU': mean,
                    'ci_lower': ci_l,
                    'ci_upper': ci_u
                })

    st_df = pd.DataFrame(st_results)
    st_df.to_csv(OUTPUT_DIR / 'spatial_temporal_breakdown.csv', index=False)

    # Print ST table
    print(f"\n{'Model':<15} | {'Spatial (Static)':>20} | {'Temporal':>20}")
    print("-" * 60)

    for model in sorted(merged_df['model'].unique()):
        row = [f"{model:<15}"]
        for st_cls in ['Spatial (Static)', 'Temporal']:
            subset = st_df[(st_df['model'] == model) & (st_df['st_class'] == st_cls)]
            if len(subset) > 0 and not pd.isna(subset['mean_mvIoU'].values[0]):
                n = subset['n_samples'].values[0]
                mean = subset['mean_mvIoU'].values[0]
                row.append(f"{mean*100:>6.1f}% (n={n:>4})")
            else:
                row.append(f"{'N/A':>20}")
        print(" | ".join(row))

    # ========== QWEN3-VL SPECIFIC BREAKDOWN ==========
    print("\n" + "="*70)
    print("QWEN3-VL CATEGORY BREAKDOWN")
    print("="*70)

    qwen3vl_entity = entity_df[entity_df['model'] == 'qwen3vl'].sort_values('mean_mvIoU', ascending=False)
    qwen3vl_st = st_df[st_df['model'] == 'qwen3vl']

    print("\nQwen3-VL by Entity Type:")
    print("-"*50)
    for _, row in qwen3vl_entity.iterrows():
        if not pd.isna(row['mean_mvIoU']):
            ci_str = f"[{row['ci_lower']*100:.1f}%, {row['ci_upper']*100:.1f}%]" if not pd.isna(row['ci_lower']) else ""
            print(f"  {row['entity_class']:<20}: {row['mean_mvIoU']*100:5.1f}% {ci_str} (n={row['n_samples']})")

    print("\nQwen3-VL by Spatial/Temporal:")
    print("-"*50)
    for _, row in qwen3vl_st.iterrows():
        if not pd.isna(row['mean_mvIoU']):
            ci_str = f"[{row['ci_lower']*100:.1f}%, {row['ci_upper']*100:.1f}%]" if not pd.isna(row['ci_lower']) else ""
            print(f"  {row['st_class']:<20}: {row['mean_mvIoU']*100:5.1f}% {ci_str} (n={row['n_samples']})")

    # ========== TOP/BOTTOM PERFORMERS BY CATEGORY ==========
    print("\n" + "="*70)
    print("TOP/BOTTOM PERFORMERS BY CATEGORY")
    print("="*70)

    # Entity type
    print("\nBy Entity Type (top performer):")
    for entity_cls in entity_classes:
        subset = entity_df[entity_df['entity_class'] == entity_cls].sort_values('mean_mvIoU', ascending=False)
        if len(subset) > 0:
            top = subset.iloc[0]
            print(f"  {entity_cls:<20}: {top['model']} ({top['mean_mvIoU']*100:.1f}%)")

    # Spatial/Temporal
    print("\nBy Spatial/Temporal (top performer):")
    for st_cls in st_classes:
        subset = st_df[st_df['st_class'] == st_cls].sort_values('mean_mvIoU', ascending=False)
        if len(subset) > 0:
            top = subset.iloc[0]
            print(f"  {st_cls:<20}: {top['model']} ({top['mean_mvIoU']*100:.1f}%)")

    # Save merged data with categories
    merged_df.to_csv(OUTPUT_DIR / 'combined_results_with_categories.csv', index=False)

    print("\n" + "="*70)
    print("FILES SAVED")
    print("="*70)
    print(f"  - {OUTPUT_DIR / 'entity_type_breakdown.csv'}")
    print(f"  - {OUTPUT_DIR / 'spatial_temporal_breakdown.csv'}")
    print(f"  - {OUTPUT_DIR / 'combined_results_with_categories.csv'}")


if __name__ == "__main__":
    main()
