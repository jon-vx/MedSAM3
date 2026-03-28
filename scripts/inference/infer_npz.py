#!/usr/bin/env python3
"""
Batch inference on images.npz / masks.npz using SAM3 + LoRA weights.

Loads all images from the NPZ files, runs inference with a text prompt,
computes IoU against ground-truth masks, and saves per-image visualizations.

Usage:
    python scripts/inference/infer_npz.py \
        --config configs/full_lora_config.yaml \
        --images images.npz \
        --masks masks.npz \
        --prompt "spleen" \
        --output-dir outputs/npz_inference
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image as PILImage

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.inference.infer_sam import SAM3LoRAInference


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute IoU between a predicted binary mask and ground-truth mask."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice coefficient between predicted and ground-truth masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2 * intersection / total)


def merge_masks(result: dict) -> np.ndarray | None:
    """Merge all predicted masks for a single prompt into one binary mask."""
    if result["num_detections"] == 0 or result["masks"] is None:
        return None
    # Union of all detection masks
    merged = result["masks"].any(axis=0)  # (H, W)
    return merged


def save_comparison(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray | None,
    iou: float,
    dice: float,
    filename: str,
    output_path: str,
):
    """Save a side-by-side comparison of GT vs predicted mask."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    # Ground truth
    axes[1].imshow(image)
    gt_overlay = np.zeros((*gt_mask.shape, 4))
    gt_overlay[gt_mask > 0] = [0, 1, 0, 0.4]
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Prediction
    axes[2].imshow(image)
    if pred_mask is not None:
        pred_overlay = np.zeros((*pred_mask.shape, 4))
        pred_overlay[pred_mask] = [1, 0, 0, 0.4]
        axes[2].imshow(pred_overlay)
    axes[2].set_title(f"Prediction (IoU={iou:.3f}, Dice={dice:.3f})")
    axes[2].axis("off")

    fig.suptitle(filename, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Batch NPZ inference with SAM3 + LoRA")
    parser.add_argument("--config", type=str, default="configs/full_lora_config.yaml",
                        help="Path to LoRA config YAML")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to LoRA weights (auto-detected if omitted)")
    parser.add_argument("--images", type=str, default="images.npz",
                        help="Path to images.npz")
    parser.add_argument("--masks", type=str, default="masks.npz",
                        help="Path to masks.npz")
    parser.add_argument("--prompt", type=str, default="spleen",
                        help="Text prompt for segmentation")
    parser.add_argument("--output-dir", type=str, default="outputs/npz_inference",
                        help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection confidence threshold")
    parser.add_argument("--nms-iou", type=float, default=0.5,
                        help="NMS IoU threshold")
    parser.add_argument("--save-images", action="store_true",
                        help="Save per-image comparison visualizations")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images to process (default: all)")
    parser.add_argument("--debug", action="store_true",
                        help="Save raw predicted masks, scores, and boxes for debugging")
    args = parser.parse_args()

    # Load NPZ data
    print(f"Loading images from {args.images}...")
    img_data = np.load(args.images, allow_pickle=True)
    images = img_data["images"]
    filenames = img_data["filenames"]

    print(f"Loading masks from {args.masks}...")
    mask_data = np.load(args.masks, allow_pickle=True)
    gt_masks = mask_data["masks"]

    n = len(images)
    if args.max_images is not None:
        n = min(n, args.max_images)
    print(f"Processing {n} images with prompt: \"{args.prompt}\"\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_images:
        vis_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    if args.debug:
        debug_dir = os.path.join(args.output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

    # Initialize model
    inferencer = SAM3LoRAInference(
        config_path=args.config,
        weights_path=args.weights,
        detection_threshold=args.threshold,
        nms_iou_threshold=args.nms_iou,
    )

    # Run inference on each image
    ious = []
    dices = []
    detection_counts = []

    for i in range(n):
        img_np = images[i]  # (300, 300, 3) uint8
        gt_mask = gt_masks[i]  # (300, 300) uint8, binary
        fname = str(filenames[i])

        pil_image = PILImage.fromarray(img_np)

        # Build datapoint and run through model (reuse internals of predict)
        datapoint = inferencer.create_datapoint(pil_image, [args.prompt])
        datapoint = inferencer.transform(datapoint)

        from sam3.train.data.collator import collate_fn_api
        from sam3.model.utils.misc import copy_data_to_device

        batch = collate_fn_api([datapoint], dict_key="input")["input"]
        batch = copy_data_to_device(batch, inferencer.device, non_blocking=True)

        with torch.no_grad():
            outputs = inferencer.model(batch)

        last_output = outputs[-1]
        pred_logits = last_output["pred_logits"]
        pred_boxes = last_output["pred_boxes"]
        pred_masks_raw = last_output.get("pred_masks", None)

        scores = pred_logits.sigmoid()[0].max(dim=-1)[0]
        keep = scores > args.threshold

        orig_h, orig_w = img_np.shape[:2]
        pred_mask = None
        num_det = 0

        if keep.sum().item() > 0 and pred_masks_raw is not None:
            from torchvision.ops import nms

            boxes_cxcywh = pred_boxes[0, keep]
            kept_scores = scores[keep]
            cx, cy, w, h = boxes_cxcywh.unbind(-1)
            x1 = (cx - w / 2) * orig_w
            y1 = (cy - h / 2) * orig_h
            x2 = (cx + w / 2) * orig_w
            y2 = (cy + h / 2) * orig_h
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

            keep_nms = nms(boxes_xyxy, kept_scores, args.nms_iou)
            num_det = len(keep_nms)

            masks_small = pred_masks_raw[0, keep][keep_nms].sigmoid() > 0.5
            masks_resized = F.interpolate(
                masks_small.unsqueeze(0).float(),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0) > 0.5  # (num_det, H, W)

            pred_mask = masks_resized.any(dim=0).cpu().numpy()  # union of all masks

        if args.debug:
            debug_path = os.path.join(debug_dir, f"{i:04d}_debug.png")
            n_panels = 3 + num_det  # image, gt, merged pred, + individual masks
            fig, axes = plt.subplots(1, max(n_panels, 3), figsize=(5 * max(n_panels, 3), 5))

            axes[0].imshow(img_np)
            axes[0].set_title(f"Image\n{fname[:30]}")
            axes[0].axis("off")

            axes[1].imshow(gt_mask, cmap="gray")
            axes[1].set_title(f"GT mask\nsum={gt_mask.sum()}")
            axes[1].axis("off")

            if pred_mask is not None:
                axes[2].imshow(pred_mask.astype(np.uint8) * 255, cmap="gray")
                axes[2].set_title(f"Merged pred\nsum={pred_mask.sum()}")
            else:
                axes[2].set_title("No detections")
            axes[2].axis("off")

            # Show individual detection masks with scores and boxes
            if num_det > 0 and pred_masks_raw is not None:
                kept_masks = masks_resized.cpu().numpy()  # (num_det, H, W)
                kept_boxes = boxes_xyxy[keep_nms].detach().cpu().numpy()
                kept_sc = kept_scores[keep_nms].detach().cpu().numpy()
                for j in range(num_det):
                    ax = axes[3 + j]
                    ax.imshow(img_np)
                    mask_overlay = np.zeros((*kept_masks[j].shape, 4))
                    mask_overlay[kept_masks[j]] = [1, 0, 0, 0.5]
                    ax.imshow(mask_overlay)
                    bx = kept_boxes[j]
                    from matplotlib.patches import Rectangle
                    rect = Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1],
                                     linewidth=2, edgecolor="cyan", facecolor="none")
                    ax.add_patch(rect)
                    ax.set_title(f"Det {j}: score={kept_sc[j]:.3f}\nbox=[{bx[0]:.0f},{bx[1]:.0f},{bx[2]:.0f},{bx[3]:.0f}]")
                    ax.axis("off")

            plt.tight_layout()
            plt.savefig(debug_path, dpi=100, bbox_inches="tight")
            plt.close()

        gt_binary = gt_mask > 0
        if pred_mask is None:
            pred_mask_binary = np.zeros_like(gt_binary)
        else:
            pred_mask_binary = pred_mask

        iou = compute_iou(pred_mask_binary, gt_binary)
        dice = compute_dice(pred_mask_binary, gt_binary)

        ious.append(iou)
        dices.append(dice)
        detection_counts.append(num_det)

        status = f"[{i+1:3d}/{n}] {fname[:40]:40s}  IoU={iou:.3f}  Dice={dice:.3f}  dets={num_det}"
        print(status)

        if args.save_images:
            out_path = os.path.join(vis_dir, f"{i:04d}.png")
            save_comparison(img_np, gt_binary, pred_mask_binary, iou, dice, fname, out_path)

    # Summary
    ious = np.array(ious)
    dices = np.array(dices)
    detection_counts = np.array(detection_counts)

    print("\n" + "=" * 60)
    print(f"Results for prompt: \"{args.prompt}\"")
    print(f"  Images processed: {n}")
    print(f"  Mean IoU:         {ious.mean():.4f} (+/- {ious.std():.4f})")
    print(f"  Median IoU:       {np.median(ious):.4f}")
    print(f"  Mean Dice:        {dices.mean():.4f} (+/- {dices.std():.4f})")
    print(f"  Median Dice:      {np.median(dices):.4f}")
    print(f"  Mean detections:  {detection_counts.mean():.1f}")
    print(f"  Images w/ dets:   {(detection_counts > 0).sum()}/{n}")
    print("=" * 60)

    # Save metrics to CSV
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w") as f:
        f.write("index,filename,iou,dice,num_detections\n")
        for i in range(n):
            f.write(f"{i},{filenames[i]},{ious[i]:.4f},{dices[i]:.4f},{detection_counts[i]}\n")
    print(f"\nMetrics saved to {csv_path}")


if __name__ == "__main__":
    main()
