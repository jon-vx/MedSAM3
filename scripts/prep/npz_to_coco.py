#!/usr/bin/env python3
"""
Convert .npz image/mask pairs to COCO format for SAM3 training pipeline.

Usage:
    python scripts/prep/npz_to_coco.py \
        --images images.npz \
        --masks masks.npz \
        --category "spleen" \
        --output data/spleen \
        --val-split 0.15
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
from skimage import measure
from tqdm import tqdm


def mask_to_coco_annotations(mask, image_id, category_id, start_ann_id):
    """
    Convert a single binary mask into COCO annotation entries.

    A single mask might contain multiple disconnected regions (e.g., two separate
    spleen fragments). Each connected component becomes its own annotation with
    its own bounding box and polygon.

    Args:
        mask:          2D numpy array, nonzero = foreground
        image_id:      which image this belongs to
        category_id:   COCO category ID
        start_ann_id:  first annotation ID to assign

    Returns:
        list of COCO annotation dicts
    """
    annotations = []
    ann_id = start_ann_id

    # Find connected components — groups of touching foreground pixels.
    # A single mask might have multiple blobs; each becomes its own annotation.
    binary = (mask > 0).astype(np.uint8)
    labeled = measure.label(binary, connectivity=2)

    for region in measure.regionprops(labeled):
        # Skip tiny regions (< 25 pixels) — likely noise from mask edges
        if region.area < 25:
            continue

        # Get bounding box: regionprops gives (min_row, min_col, max_row, max_col)
        # COCO format wants [x, y, width, height] where x,y is top-left corner
        min_row, min_col, max_row, max_col = region.bbox
        bbox_x = float(min_col)
        bbox_y = float(min_row)
        bbox_w = float(max_col - min_col)
        bbox_h = float(max_row - min_row)

        # Extract this specific connected component as its own binary mask
        component_mask = (labeled == region.label).astype(np.uint8)

        # Convert mask to polygon using marching squares (contour tracing).
        # This traces the boundary of the binary region and returns vertex coordinates.
        # level=0.5 puts the contour halfway between 0 (background) and 1 (foreground).
        contours = measure.find_contours(component_mask, level=0.5)

        polygons = []
        for contour in contours:
            # find_contours returns (row, col) pairs — flip to (x, y) for COCO
            # Then flatten: [x1, y1, x2, y2, x3, y3, ...]
            # Skip contours with < 3 points (not a valid polygon)
            if len(contour) < 3:
                continue
            # Subsample long contours to keep the JSON manageable.
            # A contour with 5000 points is overkill — every 3rd point is plenty.
            if len(contour) > 100:
                contour = contour[::3]
            polygon = []
            for row, col in contour:
                polygon.append(round(float(col), 1))  # x
                polygon.append(round(float(row), 1))  # y
            if len(polygon) >= 6:  # minimum 3 points (6 coordinates)
                polygons.append(polygon)

        if not polygons:
            continue

        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
            "area": float(region.area),
            "segmentation": polygons,
            "iscrowd": 0,
        })
        ann_id += 1

    return annotations


def convert_npz_to_coco(
    images_path,
    masks_path,
    category_name,
    output_dir,
    val_split=0.15,
    seed=42,
    images_key="images",
    masks_key="masks",
):
    """
    Full conversion pipeline: .npz files → COCO directory structure.

    Output structure (what the training script expects):
        output_dir/
        ├── train/
        │   ├── _annotations.coco.json
        │   └── image_0001.png
        │   └── image_0002.png
        │   └── ...
        └── valid/
            ├── _annotations.coco.json
            └── image_0150.png
            └── ...
    """
    output_dir = Path(output_dir)

    # ---------- Load the .npz files ----------
    # images.npz contains a single array: (N, H, W, 3) — N RGB images
    # masks.npz contains a single array: (N, H, W) — N binary masks
    print(f"Loading {images_path}...")
    images = np.load(images_path)[images_key]
    print(f"Loading {masks_path}...")
    masks = np.load(masks_path)[masks_key]

    assert len(images) == len(masks), (
        f"Mismatch: {len(images)} images vs {len(masks)} masks"
    )
    num_samples = len(images)
    print(f"Loaded {num_samples} image/mask pairs, shape: {images.shape}")

    # ---------- Shuffle and split into train/valid ----------
    # Deterministic shuffle so the split is reproducible
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    num_val = max(1, int(num_samples * val_split))
    val_indices = set(indices[:num_val].tolist())
    train_indices = set(indices[num_val:].tolist())
    print(f"Split: {len(train_indices)} train, {len(val_indices)} valid")

    # ---------- COCO category definition ----------
    # Just one category for this dataset (e.g., "spleen")
    category_id = 1
    categories = [{"id": category_id, "name": category_name, "supercategory": "organ"}]

    # ---------- Process each split ----------
    for split_name, split_indices in [("train", train_indices), ("valid", val_indices)]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        coco_images = []
        coco_annotations = []
        ann_id = 1  # COCO annotation IDs start at 1

        for i in tqdm(sorted(split_indices), desc=f"Processing {split_name}"):
            image_id = i + 1  # COCO image IDs start at 1
            filename = f"image_{i:04d}.png"

            # ------ Save image as PNG ------
            # The .npz stores uint8 RGB arrays. PIL converts to a standard image file.
            img = images[i]
            if img.dtype != np.uint8:
                # Normalize to 0-255 if stored as float
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            Image.fromarray(img).save(split_dir / filename)

            h, w = img.shape[:2]

            # ------ Add image entry to COCO JSON ------
            coco_images.append({
                "id": image_id,
                "file_name": filename,
                "width": w,
                "height": h,
            })

            # ------ Convert mask to polygon annotations ------
            mask = masks[i]
            anns = mask_to_coco_annotations(mask, image_id, category_id, ann_id)
            coco_annotations.extend(anns)
            ann_id += len(anns)

        # ------ Write the COCO JSON ------
        # This is the file the training script loads:
        #   train/_annotations.coco.json  or  valid/_annotations.coco.json
        coco_json = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories,
        }

        json_path = split_dir / "_annotations.coco.json"
        with open(json_path, "w") as f:
            json.dump(coco_json, f)

        print(f"  {split_name}: {len(coco_images)} images, {len(coco_annotations)} annotations")
        print(f"  Saved to {split_dir}")

    print(f"\nDone. Output at: {output_dir}")
    print(f"\nTo train:")
    print(f"  python scripts/training/train_sam3_lora_native.py \\")
    print(f"    --config configs/full_lora_config.yaml")
    print(f"  (set data_dir: \"{output_dir}\" in the config)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .npz image/mask pairs to COCO format"
    )
    parser.add_argument("--images", required=True, help="Path to images.npz")
    parser.add_argument("--masks", required=True, help="Path to masks.npz")
    parser.add_argument("--category", required=True, help="Category name (e.g., 'spleen')")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction for validation (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--images-key", default="images", help="Key in images.npz")
    parser.add_argument("--masks-key", default="masks", help="Key in masks.npz")

    args = parser.parse_args()

    convert_npz_to_coco(
        images_path=args.images,
        masks_path=args.masks,
        category_name=args.category,
        output_dir=args.output,
        val_split=args.val_split,
        seed=args.seed,
        images_key=args.images_key,
        masks_key=args.masks_key,
    )
