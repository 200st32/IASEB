#!/usr/bin/env python3
"""Visualize evaluation results with GT and predicted bounding boxes."""

import json
import argparse
import cv2
import numpy as np
from pathlib import Path
import math


def draw_boxes(frame, gt_box, pred_box, frame_idx):
    """Draw GT (green) and predicted (red) boxes on frame."""
    img = frame.copy()
    h, w = img.shape[:2]

    # Draw GT box (green)
    if gt_box and gt_box != [0, 0, 0, 0]:
        x1, y1, x2, y2 = [int(c) for c in gt_box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "GT", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw predicted box (red)
    if pred_box and pred_box != [0, 0, 0, 0]:
        x1, y1, x2, y2 = [int(c) for c in pred_box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, "Pred", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Add frame index
    cv2.putText(img, f"Frame {frame_idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img


def create_grid(images, cols=4):
    """Create a grid of images."""
    if not images:
        return None

    n = len(images)
    rows = math.ceil(n / cols)

    h, w = images[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = img

    return grid


def main(args):
    # Load results
    with open(args.results_path, 'r') as f:
        results = json.load(f)

    for result_idx, result in enumerate(results['results']):
        entry = result['entry']
        video_path = entry['video_path']
        gt_boxes = result['ground_truth_boxes']
        pred_boxes = result['predicted_boxes']
        caption = entry['caption']

        print(f"Processing: {caption}")
        print(f"Video: {video_path}")
        print(f"mvIoU: {result['mvIoU']:.4f}")

        # Get frame indices
        start_frame = entry.get('st_frame', 0)
        frame_step = results['evaluation_parameters']['frame_step']

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            continue

        frames_with_boxes = []
        frame_indices = []

        # Calculate which frames were sampled
        for i in range(len(gt_boxes)):
            frame_indices.append(start_frame + i * frame_step)

        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                continue

            gt_box = gt_boxes[i] if i < len(gt_boxes) else None
            pred_box = pred_boxes[i] if i < len(pred_boxes) else None

            annotated = draw_boxes(frame, gt_box, pred_box, frame_idx)
            frames_with_boxes.append(annotated)

        cap.release()

        if not frames_with_boxes:
            print("No frames to visualize")
            continue

        # Create grid
        grid = create_grid(frames_with_boxes, cols=args.cols)

        # Add caption as title
        title_height = 40
        titled_grid = np.zeros((grid.shape[0] + title_height, grid.shape[1], 3), dtype=np.uint8)
        titled_grid[title_height:] = grid

        # Truncate caption if too long
        display_caption = caption[:80] + "..." if len(caption) > 80 else caption
        cv2.putText(titled_grid, display_caption, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Add legend
        cv2.putText(titled_grid, "Green=GT  Red=Pred", (grid.shape[1] - 180, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Save
        output_path = args.output_path or f"results/viz_{Path(args.results_path).stem}_{result_idx}.png"
        cv2.imwrite(output_path, titled_grid)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--results_path", type=str, required=True,
                        help="Path to results JSON file")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output image path (default: results/viz_<name>.png)")
    parser.add_argument("--cols", type=int, default=4,
                        help="Number of columns in grid (default: 4)")

    args = parser.parse_args()
    main(args)
