# ------------------------------------------
# IASEB Evaluation Script
#
# Models: CogVLM, Ferret-V1, Shikra, Qwen3-VL
# Datasets: HC-STVG-V1, HC-STVG-V2, VidSTG, VidVRD, MeViS, RVOS
# Tasks: freeform, referral
#
# Coordinate conventions:
# - All bounding boxes converted to [xmin, ymin, xmax, ymax] internally
# - HC-STVG, VidSTG, VidVRD input format: [x, y, w, h]
# - Model outputs: normalized to 0-1000 scale
# ------------------------------------------
import argparse
import json
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import time

from .datasets import STVGDataLoader, load_dataset_config
from .utils import AverageMeter, Summary, convert_to_python_types, rescale_box_from_1000px, calculate_iou_corners
from .models import FerretSingleSample, ShikraSingleSample, CogVLMSingleSample, Qwen3VLSingleSample, MiMoVLSingleSample


# Evaluation function for HC-STVG-1&2, VidVRD, and VidSTG
def evaluate_entry(frames_with_gt, entry, runner):
    caption = entry["caption"]

    sampled_gt_boxes = [bbox for frame, bbox, frame_id in frames_with_gt]

    predicted_boxes = []
    responses = []
    predicted_frame_ids = []
    frame_inference_times = []

    for frame_idx, (frame, gt_bbox, frame_id) in enumerate(frames_with_gt):
        if frame is None:
            print(f"WARNING: None frame at index {frame_idx}, frame_id {frame_id}, video {entry.get('video_path', 'N/A')}")
            continue

        try:
            frame_H, frame_W = frame.shape[:2]
        except Exception as e:
            print(f"Error getting shape for frame_id {frame_id} even after None check. Error: {e}")
            continue
                
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        start_time = time.time()
        text, boxes, query, response = runner.run_inference(pil_image, caption)
        end_time = time.time()

        responses.append(response)

        frame_inference_times.append(end_time - start_time)
        frame_box = [0, 0, 0, 0] # Default box
        if boxes is not None and len(boxes) > 0:
            box = boxes[0].cpu().numpy().flatten()

            frame_box = rescale_box_from_1000px(box, frame_W, frame_H)
        
        predicted_boxes.append(frame_box)
        predicted_frame_ids.append(frame_id)
        
    frame_ious = [calculate_iou_corners(pred, gt) for pred, gt in zip(predicted_boxes, sampled_gt_boxes)]
    mv_iou = np.mean(frame_ious) if frame_ious else 0.0

    mv_iou_03 = np.mean([1 if iou >= 0.3 else 0 for iou in frame_ious])
    mv_iou_05 = np.mean([1 if iou >= 0.5 else 0 for iou in frame_ious])

    # Calculate timing metrics for this entry
    total_entry_inference_time = sum(frame_inference_times)
    avg_frame_inference_time = np.mean(frame_inference_times) if frame_inference_times else 0.0
    num_frames_processed = len(frames_with_gt)

    return mv_iou, mv_iou_03, mv_iou_05, predicted_boxes, total_entry_inference_time, avg_frame_inference_time, num_frames_processed, query, responses

def evaluate_entry_mevis_rvos(frames_np, entry, runner):
    caption = entry["caption"]
    predictions = {
        "pred": [],
        "pred_boxes": [],
        "pred_inference_time": [],
    }

    assert len(frames_np) == len(entry["gt_bboxs"]), \
        f"ERROR: Number of sampled frames ({len(frames_np)}) should match number of sampled GT bboxs ({len(entry['gt_bboxs'])})"
    
    for frame, (frame_id, bbox) in zip(frames_np, entry["gt_bboxs"].items()):
        frame_H, frame_W = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        start_time = time.time()
        text, boxes, prompt, response = runner.run_inference(pil_image, caption)
        end_time = time.time()

        pred_box = [0, 0, 0, 0] # Default box
        if boxes is not None and len(boxes) > 0:
            box = boxes[0].cpu().numpy().flatten()
            pred_box = rescale_box_from_1000px(box, frame_W, frame_H)

        predictions["pred"].append(response)
        predictions["pred_boxes"].append(pred_box)
        predictions["pred_inference_time"].append(end_time - start_time)

    frame_ious = [calculate_iou_corners(pred, gt) for pred, gt in zip(predictions["pred_boxes"], [bbox for frame_id, bbox in entry["gt_bboxs"].items()])]
    metrics = {
        "mv_iou": np.mean(frame_ious) if frame_ious else 0.0,
        "mv_iou03": np.mean([1 if iou >= 0.3 else 0 for iou in frame_ious]),
        "mv_iou05": np.mean([1 if iou >= 0.5 else 0 for iou in frame_ious])
    }

    return metrics, predictions, prompt


def main(args):
    if args.config:
        load_dataset_config(args.config)
    dataset = STVGDataLoader(args)
    print(f"Initializing model: {args.model}")
    if args.model == 'cogvlm':
        runner = CogVLMSingleSample()
    elif args.model == 'shikra':
        runner = ShikraSingleSample()
    elif args.model == 'ferret':
        runner = FerretSingleSample()
    elif args.model == 'qwen3vl':
        runner = Qwen3VLSingleSample()
    elif args.model == 'mimovl':
        runner = MiMoVLSingleSample()
    else:
        raise ValueError(f"Model '{args.model}' is not supported. Choose from 'cogvlm', 'shikra', 'ferret', 'qwen3vl', 'mimovl'.")
        
    mv_iou = AverageMeter('Mean Video IoU', fmt=':.4f', summary_type=Summary.AVERAGE)
    mv_iou03 = AverageMeter('Video IoU@03', fmt=':.4f', summary_type=Summary.AVERAGE)
    mv_iou05 = AverageMeter('Video IoU@05', fmt=':.4f', summary_type=Summary.AVERAGE)
    inferences_times = AverageMeter('Inference_Time', fmt=':.4f', summary_type=Summary.AVERAGE)
    predictions_list = []

    # Resume from checkpoint if exists
    processed_indices = set()
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        with open(args.checkpoint_path, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    idx = entry.get("index", -1)
                    if idx >= 0:
                        processed_indices.add(idx)
                    predictions_list.append(entry)
                    # Restore metrics from checkpoint
                    if "mvIoU" in entry:
                        mv_iou.update(entry["mvIoU"], 1)
                        mv_iou03.update(entry.get("mvIoU03", 0), 1)
                        mv_iou05.update(entry.get("mvIoU05", 0), 1)
                    elif "metrics" in entry:
                        mv_iou.update(entry["metrics"]["mv_iou"], 1)
                        mv_iou03.update(entry["metrics"]["mv_iou03"], 1)
                        mv_iou05.update(entry["metrics"]["mv_iou05"], 1)
        print(f"Resumed from checkpoint: {len(processed_indices)} samples already processed")

    overall_start_time = time.time()

    if args.entry_index > 0:
        start_index, end_index = args.entry_index, min(args.entry_index + args.max_iters, len(dataset))
    elif args.max_iters > 0:
        start_index, end_index = 0, min(args.max_iters, len(dataset))
    else:
        start_index, end_index = 0, len(dataset)

    
    for i in tqdm(range(start_index, end_index), desc="Evaluating..."):
        # Skip already processed samples (for resume)
        if i in processed_indices:
            continue

        if args.dataset == 'mevis' or args.dataset == 'rvos':
            frames, entry = dataset[i]
            assert len(frames) > 0, "no frames sampled"
            assert len(entry["caption"]) > 0, "empty caption"
            metrics, predictions, prompt = evaluate_entry_mevis_rvos(frames, entry, runner)

            mv_iou.update(metrics["mv_iou"], len(predictions["pred"]))
            mv_iou03.update(metrics["mv_iou03"], len(predictions["pred"]))
            mv_iou05.update(metrics["mv_iou05"], len(predictions["pred"]))
            inferences_times.update(np.mean(predictions["pred_inference_time"]), len(predictions["pred"]))

            result = {
                "index": i,  # For resume tracking
                "entry": entry,
                "prompt": prompt,
                "predictions": predictions,
                "metrics": metrics,
            }

        else:
            frames_with_gt, entry, gt_bboxs = dataset[i]
            assert len(frames_with_gt) > 0, "no frames sampled"
            assert len(entry["caption"]) > 0, "empty caption"
            result_mv_iou, result_mv_iou_03, result_mv_iou_05, pred_boxes, entry_time, avg_frame_time, num_frames, query, responses = evaluate_entry(frames_with_gt, entry, runner)

            pred_boxes = convert_to_python_types(pred_boxes)

            mv_iou.update(result_mv_iou, 1)
            mv_iou03.update(result_mv_iou_03, 1)
            mv_iou05.update(result_mv_iou_05, 1)
            inferences_times.update(entry_time, 1)

            result = {
                "index": i,  # For resume tracking
                "entry": entry,
                "queries": query,
                "responses": responses,
                "ground_truth_boxes": gt_bboxs,
                "predicted_boxes": pred_boxes,
                "mvIoU": float(result_mv_iou),
                "mvIoU03": float(result_mv_iou_03),
                "mvIoU05": float(result_mv_iou_05),
                "timing_info": {
                    "total_inference_time_seconds": float(entry_time),
                    "frames_processed": int(num_frames),
                    "mean_inference_time_per_frame_seconds": float(avg_frame_time)
                }
            }
        predictions_list.append(result)

        # Checkpoint after each sample
        if args.checkpoint_path:
            with open(args.checkpoint_path, "a") as f:
                f.write(json.dumps(result) + "\n")

    timing_summary = {
        "total_evaluation_time_seconds": float(time.time() - overall_start_time),
        "total_model_inference_time_seconds": float(inferences_times.sum),
        "total_samples_processed": int(len(dataset)),
        "total_frames_processed": int(mv_iou.count),
        "mean_processing_time_per_sample_seconds": float(inferences_times.avg),
        "mean_inference_time_per_frame_seconds": float(inferences_times.sum / mv_iou.count) if mv_iou.count > 0 else 0
    }

    print("\n--- Evaluation Results ---")
    print(mv_iou.summary())
    print(mv_iou03.summary())
    print(mv_iou05.summary())

    print("\n--- Timing Summary ---")
    print(f"Total Evaluation Time: {timing_summary['total_evaluation_time_seconds']:.2f} seconds")
    print(f"Total Model Inference Time: {timing_summary['total_model_inference_time_seconds']:.2f} seconds")
    print(f"Total Samples Processed: {end_index - start_index}")
    print(f"Total Frames Processed: {timing_summary['total_frames_processed']}")
    print(f"Mean Processing Time per Sample: {timing_summary['mean_processing_time_per_sample_seconds']:.4f} seconds")
    print(f"Mean Inference Time per Frame: {timing_summary['mean_inference_time_per_frame_seconds']:.4f} seconds")
    print("----------------------")

    final_output = {
        "evaluation_parameters": {
            "frame_step": args.frame_step if args.dataset != 'rvos' else 0,
            "dataset": args.dataset,
            "model": args.model,
            "task_type": args.task_type,
        },
        "timing_summary": timing_summary,
        "overall_results": {
            "avg_mviou": mv_iou.avg,
            "avg_mviou03": mv_iou03.avg,
            "avg_mviou05": mv_iou05.avg,
        },
        "results": predictions_list
    }

    if len(os.path.dirname(args.output_path)) > 0: os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(final_output, f, indent=4)
    print(f"Done! Predictions saved to {args.output_path}")

# --------------------------
# Argument Parsing and Execution
# --------------------------
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="IASEB Model Evaluation")
    parser.add_argument("--dataset", type=str, required=True,
                        help="select one from ('hcstvg1', 'hcstvg2', 'vidstg', 'vidvrd', 'mevis', 'rvos')")
    parser.add_argument("--model", type=str, required=True,
                        help="select one from ('cogvlm', 'ferret', 'shikra', 'qwen3vl')")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to dataset config YAML file (see config.example.yaml)")
    parser.add_argument("--task_type", type=str, required=False, choices=['referral', 'freeform'],
                        help="Task type i.e., 'referral', 'freeform'")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path of output JSON file.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference (default: cuda)")
    parser.add_argument("--entry_index", type=int, default=-1,
                        help="Index of a single entry to test (>= 0) or -1 for full dataset evaluation")
    parser.add_argument("--frame_step", type=int, default=5,
                        help="Frame sampling step")
    parser.add_argument("--max_iters", type=int, default=-1,
                        help="Maximum number of iterations (batches) to process (use a positive value for testing, -1 for full dataset)")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path for checkpoint file (.jsonl). Enables incremental saving and resume on restart.")

    args = parser.parse_args()
    print(args)
    main(args)

