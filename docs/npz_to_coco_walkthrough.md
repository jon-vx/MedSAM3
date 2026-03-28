# npz_to_coco.py — Code Walkthrough

Converts `.npz` image/mask pairs into COCO format for the SAM3 training pipeline.

**Script location:** `scripts/prep/npz_to_coco.py`

---

## Imports

```python
import argparse          # CLI argument parsing
import json              # Write the COCO JSON files
import numpy as np       # Load .npz arrays, array operations
from pathlib import Path # File path handling
from PIL import Image    # Save numpy arrays as PNG files
from skimage import measure  # Connected components + contour tracing
from tqdm import tqdm    # Progress bars
```

---

## `mask_to_coco_annotations()` — Core Conversion

Takes one binary mask and turns it into COCO annotation entries.

### Connected Component Labeling

```python
binary = (mask > 0).astype(np.uint8)
labeled = measure.label(binary, connectivity=2)
```

Takes the mask (where nonzero = spleen) and runs **connected component labeling**. This assigns each separate blob a unique integer. If the mask has one spleen region, `labeled` has all those pixels set to `1`. If there are two disconnected fragments, one group gets `1`, the other gets `2`.

`connectivity=2` means diagonal pixels count as connected (8-neighbor connectivity).

### Loop Over Each Region

```python
for region in measure.regionprops(labeled):
```

`regionprops` measures properties of each labeled blob — its area, bounding box, label ID, etc. We loop over each one.

### Filter Noise

```python
if region.area < 25:
    continue
```

Skip tiny blobs. Masks sometimes have stray pixels at edges from interpolation or annotation noise. 25 pixels is basically nothing.

### Bounding Box Extraction

```python
min_row, min_col, max_row, max_col = region.bbox
bbox_x = float(min_col)
bbox_y = float(min_row)
bbox_w = float(max_col - min_col)
bbox_h = float(max_row - min_row)
```

`regionprops` gives the bounding box as `(min_row, min_col, max_row, max_col)` — row-first (numpy convention). COCO wants `[x, y, width, height]` where `x,y` is the top-left corner — column-first. So `min_col` becomes `x`, `min_row` becomes `y`.

### Isolate This Component

```python
component_mask = (labeled == region.label).astype(np.uint8)
```

Isolate just this one blob. If there were 3 components in the mask, this gives a binary mask for just one of them.

### Contour Tracing (Marching Squares)

```python
contours = measure.find_contours(component_mask, level=0.5)
```

**Marching squares** — traces the boundary where the mask transitions from 0 to 1. `level=0.5` means "draw the line at the halfway point between background and foreground." Returns a list of contours (one blob can have multiple contours if it has holes).

Each contour is an array of `(row, col)` coordinates tracing the boundary.

### Subsample Long Contours

```python
if len(contour) > 100:
    contour = contour[::3]
```

A 300x300 mask can produce contours with thousands of points. That's way more precision than needed and bloats the JSON. Taking every 3rd point keeps the shape accurate enough.

### Flip to COCO Coordinate Order

```python
for row, col in contour:
    polygon.append(round(float(col), 1))  # x
    polygon.append(round(float(row), 1))  # y
```

Flip `(row, col)` to `(x, y)` — COCO uses `x,y` order. Flatten into `[x1, y1, x2, y2, x3, y3, ...]` which is what COCO's polygon segmentation format expects. Round to 1 decimal to keep file size down.

### Build the COCO Annotation Dict

```python
annotations.append({
    "id": ann_id,
    "image_id": image_id,
    "category_id": category_id,
    "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
    "area": float(region.area),
    "segmentation": polygons,
    "iscrowd": 0,
})
```

The final COCO annotation dict. `iscrowd: 0` means this is a single object instance (not a crowd region like "group of people").

This is exactly what the training script's `COCOSegmentDataset.__getitem__()` reads — it pulls `bbox`, `category_id`, and `segmentation` from each annotation.

---

## `convert_npz_to_coco()` — The Pipeline

### Load Arrays

```python
images = np.load(images_path)[images_key]
masks = np.load(masks_path)[masks_key]
```

Load the arrays. `images["images"]` gives `(208, 300, 300, 3)`, `masks["masks"]` gives `(208, 300, 300)`.

### Train/Valid Split

```python
rng = np.random.default_rng(seed)
indices = np.arange(num_samples)
rng.shuffle(indices)

num_val = max(1, int(num_samples * val_split))
val_indices = set(indices[:num_val].tolist())
train_indices = set(indices[num_val:].tolist())
```

Deterministic shuffle with seed 42. First 15% of shuffled indices go to validation, rest to training. `default_rng` is numpy's modern RNG — reproducible across runs with the same seed.

### Category Definition

```python
categories = [{"id": category_id, "name": category_name, "supercategory": "organ"}]
```

COCO requires a categories list. One entry: `{"id": 1, "name": "spleen", "supercategory": "organ"}`.

This is what gets mapped to the text prompt in training — when the dataset sees `category_id: 1`, it looks up `"spleen"` and passes that as the text prompt to SAM3.

### Image Dtype Handling

```python
if img.dtype != np.uint8:
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
```

Safety check. If images are stored as floats 0.0–1.0 instead of uint8 0–255, scale them up. PIL needs uint8 to save as PNG.

### Save Image to Disk

```python
Image.fromarray(img).save(split_dir / filename)
```

Numpy array → PIL Image → PNG file on disk. This is what `PILImage.open(img_path).convert("RGB")` in the training script loads back.

### Write the COCO JSON

```python
coco_json = {
    "images": coco_images,
    "annotations": coco_annotations,
    "categories": categories,
}
json_path = split_dir / "_annotations.coco.json"
```

Assembles the three required top-level keys of a COCO JSON and writes it. The filename `_annotations.coco.json` is hardcoded in `COCOSegmentDataset.__init__()` at line 123 of the training script — it must be exactly this name.

---

## Output Structure

```
data/spleen/
├── train/
│   ├── _annotations.coco.json
│   ├── image_0000.png
│   ├── image_0001.png
│   └── ...
└── valid/
    ├── _annotations.coco.json
    ├── image_0003.png
    └── ...
```

## Usage

```bash
python scripts/prep/npz_to_coco.py \
    --images images.npz \
    --masks masks.npz \
    --category "spleen" \
    --output data/spleen \
    --val-split 0.15
```

Then set `data_dir: "data/spleen"` in your training config and run training as normal.
