#!/usr/bin/env python3
"""Aggregate results from array job outputs into a single JSON file."""

import json
import argparse
from pathlib import Path
import glob


def aggregate_results(pattern: str, output_path: str):
    """Aggregate multiple result JSON files into one."""

    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return

    print(f"Found {len(files)} result files")

    # Initialize aggregated structure
    aggregated = {
        "evaluation_parameters": None,
        "timing_summary": {
            "total_evaluation_time_seconds": 0,
            "total_model_inference_time_seconds": 0,
            "total_samples_processed": 0,
            "total_frames_processed": 0,
        },
        "overall_results": {
            "total_iou_sum": 0,
            "total_iou03_sum": 0,
            "total_iou05_sum": 0,
            "total_count": 0,
        },
        "results": []
    }

    for filepath in files:
        print(f"Processing: {filepath}")
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ERROR reading {filepath}: {e}")
            continue

        # Copy evaluation parameters from first file
        if aggregated["evaluation_parameters"] is None:
            aggregated["evaluation_parameters"] = data.get("evaluation_parameters", {})

        # Aggregate timing
        timing = data.get("timing_summary", {})
        aggregated["timing_summary"]["total_evaluation_time_seconds"] += timing.get("total_evaluation_time_seconds", 0)
        aggregated["timing_summary"]["total_model_inference_time_seconds"] += timing.get("total_model_inference_time_seconds", 0)
        aggregated["timing_summary"]["total_samples_processed"] += timing.get("total_samples_processed", 0)
        aggregated["timing_summary"]["total_frames_processed"] += timing.get("total_frames_processed", 0)

        # Aggregate results
        results = data.get("results", [])
        aggregated["results"].extend(results)

        # Sum IoU values for averaging later
        overall = data.get("overall_results", {})
        n_samples = len(results)
        if n_samples > 0:
            aggregated["overall_results"]["total_iou_sum"] += overall.get("avg_mviou", 0) * n_samples
            aggregated["overall_results"]["total_iou03_sum"] += overall.get("avg_mviou03", 0) * n_samples
            aggregated["overall_results"]["total_iou05_sum"] += overall.get("avg_mviou05", 0) * n_samples
            aggregated["overall_results"]["total_count"] += n_samples

    # Compute final averages
    total_count = aggregated["overall_results"]["total_count"]
    if total_count > 0:
        aggregated["overall_results"] = {
            "avg_mviou": aggregated["overall_results"]["total_iou_sum"] / total_count,
            "avg_mviou03": aggregated["overall_results"]["total_iou03_sum"] / total_count,
            "avg_mviou05": aggregated["overall_results"]["total_iou05_sum"] / total_count,
            "total_samples": total_count,
        }

    # Compute timing averages
    if aggregated["timing_summary"]["total_samples_processed"] > 0:
        total_samples = aggregated["timing_summary"]["total_samples_processed"]
        total_inference = aggregated["timing_summary"]["total_model_inference_time_seconds"]
        aggregated["timing_summary"]["mean_inference_time_per_sample_seconds"] = total_inference / total_samples

    # Save aggregated results
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nAggregated {len(aggregated['results'])} samples")
    print(f"Overall Results:")
    print(f"  avg_mviou: {aggregated['overall_results'].get('avg_mviou', 0):.4f}")
    print(f"  avg_mviou03: {aggregated['overall_results'].get('avg_mviou03', 0):.4f}")
    print(f"  avg_mviou05: {aggregated['overall_results'].get('avg_mviou05', 0):.4f}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate array job results")
    parser.add_argument("--pattern", type=str, required=True,
                        help="Glob pattern for result files (e.g., 'results/qwen3vl_vidvrd_323000_*.json')")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for aggregated results")

    args = parser.parse_args()
    aggregate_results(args.pattern, args.output)
