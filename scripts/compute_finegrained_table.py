#!/usr/bin/env python3
"""
Compute entity-class and fine-grained interaction-type performance for all models.

Outputs two tables:
  1. Entity-level (L2): Human-Human, Human-Object, etc.
  2. Fine-grained (L4): Affective, Antagonistic, ..., Supportive (14 paper categories)

Data sources:
  - 9 original models: _archive/adaw_eval_results_coarse_only.csv
  - Qwen3-VL, MiMo-VL: results/ (.json + .jsonl checkpoints)
  - Fine-grained classification: _archive/vg12_vrd_stg_mevis_rvos_gpt4omini_entity_class_uniq_captions_v17.json

Follows the same aggregation workflow as scripts/analysis/10_categorize_tab1.py:
  - Multi-label categories from v17 are expanded (one row per label per sample)
  - "Spatial" + "Movement" → averaged into "Relational Movement"
  - "Physical Interaction" → renamed to "Physical"
"""

import json
import glob
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COARSE_CSV = os.path.join(BASE_DIR, "_archive/adaw_eval_results_coarse_only.csv")
V17_JSON = os.path.join(BASE_DIR, "_archive/vg12_vrd_stg_mevis_rvos_gpt4omini_entity_class_uniq_captions_v17.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# The 14 paper categories (after merging Spatial+Movement and renaming Physical Interaction)
PAPER_CATEGORIES = [
    "Affective", "Antagonistic", "Body Motion", "Communicative",
    "Competitive", "Cooperative", "Relational Movement", "Observation",
    "Passive", "Physical", "Provisioning", "Proximity", "Social", "Supportive"
]

ENTITY_CLASSES = [
    "Animal-Animal", "Animal-Object", "Human-Animal", "Human-Human",
    "Human-Object", "Human-Self", "No Interaction", "Object-Object"
]


def norm(c):
    if pd.isna(c):
        return ""
    return str(c).strip().lower().replace(".", "").replace('"', "").replace("\\", "")


def load_quality_filters():
    vrd_ff = set(e["ID"] for e in json.load(open(os.path.join(BASE_DIR, "vidvrd_filtered_freeform_subset.json"))))
    vrd_r = set(e["ID"] for e in json.load(open(os.path.join(BASE_DIR, "vidvrd_filtered_referral_subset.json"))))
    stg_ff_raw = json.load(open(os.path.join(BASE_DIR, "vidstg_filtered_freeform_subset.json")))
    stg_ff = set((e["video_path"], e["target_id"], norm(e["caption"])) for e in stg_ff_raw)
    stg_r_raw = json.load(open(os.path.join(BASE_DIR, "vidstg_filtered_referral_subset.json")))
    stg_r = set((e["video_path"], e["target_id"], norm(e["caption"])) for e in stg_r_raw)
    return {"vidvrd_freeform": vrd_ff, "vidvrd_referral": vrd_r,
            "vidstg_freeform": stg_ff, "vidstg_referral": stg_r}


def should_keep(entry, dataset, task, filters):
    if dataset == "vidvrd":
        return entry.get("ID") in filters[f"vidvrd_{task}"]
    if dataset == "vidstg":
        return (entry.get("video_path", ""), entry.get("target_id", ""),
                norm(entry.get("caption", ""))) in filters[f"vidstg_{task}"]
    return True


def parse_filename(fname, model):
    base = fname.replace(f"{model}_", "", 1)
    if base.startswith("referral_"):
        task = "referral"
        dataset = base.replace("referral_", "").split("_")[0]
    else:
        task = "freeform"
        dataset = base.split("_")[0]
    return dataset, task


def load_new_model_results(model, filters):
    """Load results for a new model from .json/.jsonl files + combined CSV."""
    records = []
    seen = set()

    combined_csv = os.path.join(RESULTS_DIR, "combined_results_with_qwen3vl.csv")
    if os.path.exists(combined_csv):
        cdf = pd.read_csv(combined_csv)
        cdf = cdf[cdf["model"] == model]
        for _, row in cdf.iterrows():
            caption = norm(row.get("caption", ""))
            key = (row["dataset"], row["task"], caption, str(row.get("video_path", "")), "")
            if key not in seen:
                seen.add(key)
                records.append({
                    "model": model, "dataset": row["dataset"], "task": row["task"],
                    "caption": row.get("caption", ""), "caption_norm": caption,
                    "mvIoU": row["mvIoU"],
                })

    for fp in sorted(glob.glob(os.path.join(RESULTS_DIR, f"{model}_*.json"))):
        fname = os.path.basename(fp)
        dataset, task = parse_filename(fname, model)
        try:
            data = json.load(open(fp))
        except (json.JSONDecodeError, IOError):
            continue
        for r in data.get("results", []):
            entry = r.get("entry", r)
            if not should_keep(entry, dataset, task, filters):
                continue
            caption_raw = entry.get("caption", "")
            caption = norm(caption_raw)
            key = (dataset, task, caption, entry.get("video_path", ""), str(entry.get("target_id", "")))
            if key not in seen:
                seen.add(key)
                records.append({
                    "model": model, "dataset": dataset, "task": task,
                    "caption": caption_raw, "caption_norm": caption,
                    "mvIoU": r.get("mvIoU", r.get("mvIoU_tube_step")),
                })

    for fp in sorted(glob.glob(os.path.join(RESULTS_DIR, f"{model}_*.jsonl"))):
        fname = os.path.basename(fp)
        dataset, task = parse_filename(fname, model)
        json_fp = fp.replace(".jsonl", ".json")
        if os.path.exists(json_fp):
            continue
        try:
            with open(fp) as f:
                for line in f:
                    r = json.loads(line)
                    entry = r.get("entry", r)
                    if not should_keep(entry, dataset, task, filters):
                        continue
                    caption_raw = entry.get("caption", "")
                    caption = norm(caption_raw)
                    key = (dataset, task, caption, entry.get("video_path", ""), str(entry.get("target_id", "")))
                    if key not in seen:
                        seen.add(key)
                        records.append({
                            "model": model, "dataset": dataset, "task": task,
                            "caption": caption_raw, "caption_norm": caption,
                            "mvIoU": r.get("mvIoU", r.get("mvIoU_tube_step")),
                        })
        except (json.JSONDecodeError, IOError):
            continue

    return pd.DataFrame(records) if records else pd.DataFrame()


def load_v17_lookup():
    """Load v17 classification: caption_norm -> list of fine-grained categories."""
    data = json.load(open(V17_JSON))
    lookup = {}
    for entry in data:
        key = norm(entry["caption"])
        cats = entry.get("category", [])
        if isinstance(cats, str):
            cats = [cats]
        lookup[key] = cats
    return lookup


def build_all_samples():
    """Build a single DataFrame with all per-sample results for all models."""
    print("Loading quality filters...")
    filters = load_quality_filters()

    # 9 original models from coarse CSV
    print(f"Loading 9 original models from {COARSE_CSV}...")
    coarse = pd.read_csv(COARSE_CSV, usecols=[
        "model", "dataset", "task", "caption", "mvIoU",
        "entity_level0_cls", "st_level0_cls"
    ])
    coarse["caption_norm"] = coarse["caption"].apply(norm)
    print(f"  {len(coarse)} rows, models: {sorted(coarse['model'].unique())}")

    # New models
    new_dfs = []
    for model in ["qwen3vl", "mimovl"]:
        print(f"Loading {model} results...")
        df = load_new_model_results(model, filters)
        if len(df) > 0:
            # Merge entity/ST categories from coarse CSV lookup
            cat_lookup = coarse.groupby(["caption_norm", "dataset"]).first()[
                ["st_level0_cls", "entity_level0_cls"]
            ].reset_index()
            df = df.merge(cat_lookup, on=["caption_norm", "dataset"], how="left")
            print(f"  {len(df)} samples, category coverage: {df['entity_level0_cls'].notna().mean()*100:.0f}%")
            new_dfs.append(df)

    # Combine all
    cols = ["model", "dataset", "task", "caption", "caption_norm", "mvIoU",
            "entity_level0_cls", "st_level0_cls"]
    all_df = pd.concat([coarse[cols]] + [d[cols] for d in new_dfs], ignore_index=True)
    print(f"\nTotal samples: {len(all_df)}, Models: {sorted(all_df['model'].unique())}")
    return all_df


def compute_entity_table(all_df):
    """Compute entity-level (L2) performance: mean mvIoU per model per entity class."""
    rows = []
    for model in sorted(all_df["model"].unique()):
        mdf = all_df[all_df["model"] == model]
        row = {"model": model, "n": len(mdf)}
        for cls in ENTITY_CLASSES:
            sub = mdf[mdf["entity_level0_cls"] == cls]["mvIoU"].dropna()
            row[cls] = round(sub.mean() * 100, 1) if len(sub) > 0 else ""
        rows.append(row)
    return pd.DataFrame(rows)


def compute_finegrained_table(all_df, v17_lookup):
    """Compute fine-grained (L4) performance using v17 multi-label categories."""
    # Assign v17 categories to each sample
    all_df = all_df.copy()
    all_df["v17_cats"] = all_df["caption_norm"].map(v17_lookup)

    matched = all_df["v17_cats"].notna().sum()
    total = len(all_df)
    print(f"\nv17 category match: {matched}/{total} ({matched/total*100:.1f}%)")

    # Expand multi-label: one row per (sample, category)
    expanded_rows = []
    for _, row in all_df.iterrows():
        cats = row["v17_cats"]
        if not isinstance(cats, list):
            continue
        for cat in cats:
            expanded_rows.append({
                "model": row["model"],
                "mvIoU": row["mvIoU"],
                "category": cat,
            })

    expanded = pd.DataFrame(expanded_rows)
    print(f"Expanded to {len(expanded)} (model, sample, category) rows")

    # Pivot: mean mvIoU per (model, category)
    pivot = expanded.groupby(["model", "category"])["mvIoU"].mean().unstack(fill_value=np.nan)

    # Merge Spatial + Movement → Relational Movement (average of the two)
    if "Spatial" in pivot.columns and "Movement" in pivot.columns:
        pivot["Relational Movement"] = pivot[["Spatial", "Movement"]].mean(axis=1)
        pivot = pivot.drop(columns=["Spatial", "Movement"])
    elif "Movement" in pivot.columns:
        pivot = pivot.rename(columns={"Movement": "Relational Movement"})
    elif "Spatial" in pivot.columns:
        pivot = pivot.rename(columns={"Spatial": "Relational Movement"})

    # Rename Physical Interaction → Physical
    if "Physical Interaction" in pivot.columns:
        pivot = pivot.rename(columns={"Physical Interaction": "Physical"})

    # Scale to percentage and round
    pivot = (pivot * 100).round(1)

    # Reorder columns to match paper order
    ordered_cols = [c for c in PAPER_CATEGORIES if c in pivot.columns]
    extra_cols = [c for c in pivot.columns if c not in PAPER_CATEGORIES]
    if extra_cols:
        print(f"  Warning: unexpected categories: {extra_cols}")
    pivot = pivot[ordered_cols + extra_cols]

    # Reset index for clean output
    pivot = pivot.reset_index().rename(columns={"index": "model"})
    return pivot


def main():
    all_df = build_all_samples()

    print("\nLoading v17 fine-grained classifications...")
    v17_lookup = load_v17_lookup()
    print(f"  {len(v17_lookup)} unique caption entries")

    # Entity-level table
    print("\n" + "=" * 70)
    print("ENTITY-LEVEL (L2) PERFORMANCE")
    print("=" * 70)
    entity_table = compute_entity_table(all_df)
    print(entity_table.to_string(index=False))
    entity_out = os.path.join(RESULTS_DIR, "entity_level_performance.csv")
    entity_table.to_csv(entity_out, index=False)
    print(f"\nSaved: {entity_out}")

    # Fine-grained table
    print("\n" + "=" * 70)
    print("FINE-GRAINED (L4) PERFORMANCE - 14 Paper Categories")
    print("=" * 70)
    fg_table = compute_finegrained_table(all_df, v17_lookup)
    print(fg_table.to_string(index=False))
    fg_out = os.path.join(RESULTS_DIR, "finegrained_performance.csv")
    fg_table.to_csv(fg_out, index=False)
    print(f"\nSaved: {fg_out}")


if __name__ == "__main__":
    main()
