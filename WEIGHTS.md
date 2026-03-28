# Downloading and Managing Weights

## Base SAM3 Model Weights

The base SAM3 model (~3GB) is automatically downloaded from HuggingFace when you first run training or inference. You need to authenticate first:

```bash
pip install huggingface_hub
huggingface-cli login
```

All configs reference the model as `facebook/sam3`. By default, weights are cached in `~/.cache/huggingface/`. You can override this by setting `model.cache_dir` in your config.

## LoRA Weights (Trained)

Training produces two weight files:

| File | Description |
|------|-------------|
| `best_lora_weights.pt` | Best checkpoint by validation metric |
| `last_lora_weights.pt` | Most recent checkpoint |

### Output Paths by Config

Each config saves LoRA weights to a different directory:

| Config | Output Directory |
|--------|-----------------|
| `configs/base_config.yaml` | `outputs/sam3_lora/` |
| `configs/full_lora_config.yaml` | `outputs/sam3_lora_full/` |
| `configs/minimal_lora_config.yaml` | `outputs/sam3_lora_minimal/` |
| `configs/light_lora_config.yaml` | `outputs/sam3_lora_light/` |
| `configs/crack_detection_config.yaml` | `outputs/crack_detection_lora/` |
| `configs/sam3_lora_standalone.yaml` | `checkpoints/` |
| `sam3_lora_configs/lora_base.yaml` (Hydra) | `/workspace/outputs/sam3_lora_minimal/checkpoints/` |

### Directory Structure

After training, your output directory will look like:

```
outputs/sam3_lora_full/
├── best_lora_weights.pt    # ~10-50MB depending on config tier
├── last_lora_weights.pt
└── logs/
```

## Using Weights for Inference

The inference script auto-detects weights from the config's `output.output_dir`:

```bash
# Auto-detects best_lora_weights.pt from config's output_dir
python scripts/inference/infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image path/to/image.jpg \
  --prompt "skin lesion"

# Or specify weights explicitly
python scripts/inference/infer_sam.py \
  --config configs/full_lora_config.yaml \
  --weights outputs/sam3_lora_full/best_lora_weights.pt \
  --image path/to/image.jpg \
  --prompt "skin lesion"
```

## Sharing Weights

LoRA weights are small (10-50MB vs 3GB for the full model) and safe to distribute separately. The `outputs/` directory is gitignored, so use external storage for sharing:

- **HuggingFace Hub**: Set `output.push_to_hub: true` and `output.hub_model_id` in your config
- **Manual upload**: Upload the `.pt` file to Google Drive, S3, etc.

To use shared weights, download the `.pt` file and pass it via `--weights` at inference time.
