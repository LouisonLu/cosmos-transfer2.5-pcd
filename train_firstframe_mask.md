# Masked First-Frame PCD Training

This recipe trains Cosmos Transfer2.5 with:

```text
input  = masked first RGB frame + PCD video + prompt
target = full RGB 360 video
```

The dataset and checkpoints stay outside the repository.

## Data contract

```text
<dataset-root>/
├── rgb_videos/<stem>.mp4
├── pcd_videos/<stem>_pcd.mp4
└── captions/<stem>.json
```

The caption JSON must contain a non-empty `caption`, `text`, or `prompt` field.
The RGB and PCD videos must have matching stems and compatible frame geometry.

## Setup

```bash
git lfs install
uv sync --extra=cu128
source .venv/bin/activate
uv run --with huggingface-hub hf auth login
wandb login
```

Use `--extra=cu130` for a CUDA 13 environment. Keep credentials in the provider
credential store or outside the repository; never put them in this document.

## Variables

```bash
export REPO_ROOT="$(pwd)"
export DATASET_ROOT="/path/to/pcd-dataset"
export OUTPUT_ROOT="/path/to/cosmos-output"
export MASK_PATH="/path/to/mask.png"
export NPROC=8
export MASTER_PORT=12345
```

## Built-in mask training

The built-in Waymo mask uses the masked dataset loader and PCD as the depth input:

```bash
cd "$REPO_ROOT"
source .venv/bin/activate
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"

torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" -m scripts.train \
  --config=cosmos_transfer2/singleview_mask_config.py \
  -- \
  data_train=example_singleview_train_data_depth_mask \
  experiment=transfer2_singleview_posttrain_pcd_rgb_image_context_example \
  dataloader_train.dataset.dataset_dir="$DATASET_ROOT" \
  'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
  dataloader_train.dataset.input_video_dir=pcd_videos \
  dataloader_train.dataset.target_video_dir=rgb_videos \
  dataloader_train.dataset.input_video_suffix=_pcd \
  dataloader_train.dataset.control_video_dir_override=pcd_videos \
  dataloader_train.dataset.control_video_suffix=_pcd \
  dataloader_train.dataset.use_image_context=True \
  dataloader_train.dataset.image_context_from_rgb_first_frame=True \
  dataloader_train.dataset.mask_image_context=True \
  dataloader_train.dataset.mask_mode=waymo \
  model.config.freeze_base_model=False \
  model.config.hint_keys=depth \
  model.config.fsdp_shard_size="$NPROC" \
  optimizer.lr=1e-4 \
  trainer.max_iter=2000 \
  checkpoint.save_iter=500
```

## Custom PNG mask training

Replace the `mask_mode` override with the path to a binary PNG:

```bash
  dataloader_train.dataset.image_context_mask_reference_path="$MASK_PATH"
```

The PNG path is intentionally supplied at runtime. Do not commit the mask or a
machine-specific path.

## Resume and convert

Add these overrides to resume from an existing checkpoint:

```bash
  checkpoint.load_path="/path/to/checkpoints/iter_000001500" \
  checkpoint.load_training_state=True \
  checkpoint.strict_resume=True
```

Convert a DCP checkpoint using a path outside the repository:

```bash
python scripts/convert_distcp_to_pt.py \
  "/path/to/checkpoints/iter_000002000/model" \
  "/path/to/checkpoints/iter_000002000"
```

The generated `model_ema_bf16.pt` should also remain outside the source checkout.
