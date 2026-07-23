# Train First-Frame Mask

This note is for the training setup where:

- `X = masked first RGB frame + PCD video + prompt`
- `Y = full RGB 360 video`

The masked first frame uses:

- `mask_image_context=True`
- built-in mask: `mask_mode="waymo"`
- or custom PNG: `image_context_mask_reference_path="/path/to/mask.png"`

This is the code path that matches your current intended setup.

## 1. What this route actually uses

Config wrapper:

- `cosmos_transfer2/singleview_mask_config.py`

Experiment:

- `transfer2_singleview_posttrain_pcd_rgb_image_context_example`

Important extra override:

- `data_train=example_singleview_train_data_depth_mask`

Why this matters:

- `singleview_mask_config.py` registers the masked dataset loader
- `example_singleview_train_data_depth_mask` uses `SingleViewTransferDatasetMask`
- that dataset loader is where `mask_image_context=True` and either a built-in mask or a custom mask PNG actually take effect

If you do not switch `data_train` to `example_singleview_train_data_depth_mask`, you may still be using the non-mask dataset loader.

## 2. Hardware

Recommended from the original notes:

- total VRAM: at least `500 GB`
- disk: at least `1 TB`

Typical working setup:

- `8 x 80 GB` GPUs or better
- `8 x 96 GB` GPUs is even safer

## 3. Expected dataset layout

Training dataset root:

```text
datasets/cosmos_train_2000/
├── rgb_videos/
├── pcd_videos/
└── captions/
```

File matching rule:

- `rgb_videos/<video_id>.mp4`
- `pcd_videos/<video_id>_vis_pcd.mp4`
- `captions/<video_id>.json`

The caption JSON should contain a prompt field that the loader can read.

## 4. Fresh machine setup

### 4.1 System packages

```bash
apt-get update
apt-get install -y git git-lfs ffmpeg tmux rclone
```

### 4.2 GitHub login and clone

Use a GitHub username + token because the repo is private.

```bash
export GITHUB_USER="your_github_username"
export GITHUB_TOKEN="your_github_token"
export REPO_URL="https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/LouisonLu/cosmos-transfer2.5-pcd.git"

cd /workspace
git clone "$REPO_URL"
cd /workspace/cosmos-transfer2.5-pcd
```

Optional if LFS files are needed:

```bash
git lfs install
git lfs pull
```

### 4.3 Python environment

```bash
uv sync --extra=cu128
source .venv/bin/activate
```

### 4.4 Extra Python tools

```bash
pip install wandb huggingface-hub
```

### 4.5 Hugging Face login

```bash
huggingface-cli login
```

### 4.6 W&B login

```bash
wandb login
```

### 4.7 Dropbox / rclone config

```bash
rclone config
```

### 4.8 Basic verification

```bash
python -c "import cosmos_oss; import torch; import wandb; print('cosmos_oss OK'); print(torch.__version__); print('wandb OK')"
nvidia-smi
```

## 5. Download training data from Dropbox

These are the expected remotes for the 2000-sample dataset:

```bash
export DATASET_ROOT="/workspace/cosmos-transfer2.5-pcd/datasets/cosmos_train_2000"

export RGB_REMOTE="dropbox:/World Model (cosmos)/360 video/cosmos_train_2000/rgb_videos"
export PCD_REMOTE="dropbox:/World Model (cosmos)/360 video/cosmos_train_2000/pcd_videos"
export CAPTION_REMOTE="dropbox:/World Model (cosmos)/360 video/cosmos_train_2000/captions"

mkdir -p "$DATASET_ROOT/rgb_videos" "$DATASET_ROOT/pcd_videos" "$DATASET_ROOT/captions"
```

Download RGB:

```bash
rclone copy \
  "$RGB_REMOTE" \
  "$DATASET_ROOT/rgb_videos" \
  --progress \
  --transfers 4 \
  --checkers 8
```

Download PCD:

```bash
rclone copy \
  "$PCD_REMOTE" \
  "$DATASET_ROOT/pcd_videos" \
  --progress \
  --transfers 4 \
  --checkers 8
```

Download captions:

```bash
rclone copy \
  "$CAPTION_REMOTE" \
  "$DATASET_ROOT/captions" \
  --progress \
  --transfers 4 \
  --checkers 8
```

Quick count check:

```bash
find "$DATASET_ROOT/rgb_videos" -type f -name '*.mp4' | wc -l
find "$DATASET_ROOT/pcd_videos" -type f -name '*.mp4' | wc -l
find "$DATASET_ROOT/captions" -type f -name '*.json' | wc -l
```

## 6. Main training command

This is the command for your intended setup:

- masked first RGB frame
- `mask_mode="waymo"`
- PCD video as input/control
- RGB video as target
- caption JSON as prompt

```bash
cd /workspace/cosmos-transfer2.5-pcd
source .venv/bin/activate

export IMAGINAIRE_OUTPUT_ROOT=/tmp/imaginaire4-output

torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
  --config=cosmos_transfer2/singleview_mask_config.py \
  -- \
  data_train=example_singleview_train_data_depth_mask \
  experiment=transfer2_singleview_posttrain_pcd_rgb_image_context_example \
  dataloader_train.dataset.dataset_dir="/workspace/cosmos-transfer2.5-pcd/datasets/cosmos_train_2000" \
  'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
  dataloader_train.dataset.input_video_dir="pcd_videos" \
  dataloader_train.dataset.target_video_dir="rgb_videos" \
  dataloader_train.dataset.input_video_suffix="_vis_pcd" \
  dataloader_train.dataset.control_video_dir_override="pcd_videos" \
  dataloader_train.dataset.control_video_suffix="_vis_pcd" \
  dataloader_train.dataset.use_image_context=True \
  dataloader_train.dataset.image_context_from_rgb_first_frame=True \
  dataloader_train.dataset.mask_image_context=True \
  dataloader_train.dataset.mask_mode="waymo" \
  model.config.freeze_base_model=False \
  model.config.hint_keys=depth \
  model.config.fsdp_shard_size=8 \
  optimizer.lr=1e-4 \
  trainer.max_iter=2000 \
  checkpoint.save_iter=500
```

If you want to use your own mask PNG directly instead of a built-in shape, keep:

- `dataloader_train.dataset.mask_image_context=True`
- `dataloader_train.dataset.image_context_mask_reference_path="/path/to/mask.png"`

When `image_context_mask_reference_path` is set, the dataset loader uses that PNG directly. In that case `mask_mode` is optional.

```bash
cd /workspace/cosmos-transfer2.5-pcd
source .venv/bin/activate

export IMAGINAIRE_OUTPUT_ROOT=/tmp/imaginaire4-output

torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
  --config=cosmos_transfer2/singleview_mask_config.py \
  -- \
  data_train=example_singleview_train_data_depth_mask \
  experiment=transfer2_singleview_posttrain_pcd_rgb_image_context_example \
  dataloader_train.dataset.dataset_dir="/workspace/cosmos-transfer2.5-pcd/datasets/cosmos_train_2000" \
  'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
  dataloader_train.dataset.input_video_dir="pcd_videos" \
  dataloader_train.dataset.target_video_dir="rgb_videos" \
  dataloader_train.dataset.input_video_suffix="_vis_pcd" \
  dataloader_train.dataset.control_video_dir_override="pcd_videos" \
  dataloader_train.dataset.control_video_suffix="_vis_pcd" \
  dataloader_train.dataset.use_image_context=True \
  dataloader_train.dataset.image_context_from_rgb_first_frame=True \
  dataloader_train.dataset.mask_image_context=True \
  dataloader_train.dataset.image_context_mask_reference_path="/workspace/cosmos-transfer2.5-pcd/masks/mask_common_1024x512.png" \
  model.config.freeze_base_model=False \
  model.config.hint_keys=depth \
  model.config.fsdp_shard_size=8 \
  optimizer.lr=1e-4 \
  trainer.max_iter=2000 \
  checkpoint.save_iter=500
```

## 7. Resume training

Resume from iteration 1500:

```bash
cd /workspace/cosmos-transfer2.5-pcd
source .venv/bin/activate

export IMAGINAIRE_OUTPUT_ROOT=/tmp/imaginaire4-output

torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
  --config=cosmos_transfer2/singleview_mask_config.py \
  -- \
  data_train=example_singleview_train_data_depth_mask \
  experiment=transfer2_singleview_posttrain_pcd_rgb_image_context_example \
  dataloader_train.dataset.dataset_dir="/workspace/cosmos-transfer2.5-pcd/datasets/cosmos_train_2000" \
  'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
  dataloader_train.dataset.input_video_dir="pcd_videos" \
  dataloader_train.dataset.target_video_dir="rgb_videos" \
  dataloader_train.dataset.input_video_suffix="_vis_pcd" \
  dataloader_train.dataset.control_video_dir_override="pcd_videos" \
  dataloader_train.dataset.control_video_suffix="_vis_pcd" \
  dataloader_train.dataset.use_image_context=True \
  dataloader_train.dataset.image_context_from_rgb_first_frame=True \
  dataloader_train.dataset.mask_image_context=True \
  dataloader_train.dataset.mask_mode="waymo" \
  model.config.freeze_base_model=False \
  model.config.hint_keys=depth \
  model.config.fsdp_shard_size=8 \
  optimizer.lr=1e-4 \
  trainer.max_iter=2000 \
  checkpoint.save_iter=500 \
  checkpoint.load_path=/tmp/imaginaire4-output/cosmos_transfer2_posttrain/local_single_view/transfer2_singleview_posttrain_pcd_rgb_image_context_example/checkpoints/iter_000001500 \
  checkpoint.load_training_state=True \
  checkpoint.strict_resume=True
```

## 8. Convert DCP checkpoint to PyTorch checkpoint

Example for iteration 2000:

```bash
cd /workspace/cosmos-transfer2.5-pcd
source .venv/bin/activate

CHECKPOINT_DIR=/tmp/imaginaire4-output/cosmos_transfer2_posttrain/local_single_view/transfer2_singleview_posttrain_pcd_rgb_image_context_example/checkpoints/iter_000002000

python scripts/convert_distcp_to_pt.py \
  "$CHECKPOINT_DIR/model" \
  "$CHECKPOINT_DIR"
```

Expected useful output:

- `model_ema_bf16.pt`

## 9. Upload converted checkpoint to Hugging Face

If you want to publish the converted checkpoint:

```bash
export HF_REPO="LouisonLu/cosmos-transfer2.5-pcd-firstframe-mask-waymo-iter2000"
export CHECKPOINT_DIR="/tmp/imaginaire4-output/cosmos_transfer2_posttrain/local_single_view/transfer2_singleview_posttrain_pcd_rgb_image_context_example/checkpoints/iter_000002000"

huggingface-cli upload "$HF_REPO" \
  "$CHECKPOINT_DIR/model_ema_bf16.pt" \
  iter_000002000/model_ema_bf16.pt \
  --repo-type model
```

## 10. Upload checkpoints to Dropbox

If you also want a Dropbox copy:

```bash
export CKPT_REMOTE="dropbox:/World Model (cosmos)/360 video/cosmos_train_2000/checkpoints/firstframe_mask_waymo_iter2000"
export CHECKPOINT_DIR="/tmp/imaginaire4-output/cosmos_transfer2_posttrain/local_single_view/transfer2_singleview_posttrain_pcd_rgb_image_context_example/checkpoints/iter_000002000"

rclone copy \
  "$CHECKPOINT_DIR" \
  "$CKPT_REMOTE" \
  --progress \
  --transfers 4 \
  --checkers 8
```

## 11. Upload training logs if needed

```bash
export LOG_REMOTE="dropbox:/World Model (cosmos)/360 video/cosmos_train_2000/logs/firstframe_mask_waymo"

rclone copy \
  "/tmp/imaginaire4-output/cosmos_transfer2_posttrain/local_single_view/transfer2_singleview_posttrain_pcd_rgb_image_context_example" \
  "$LOG_REMOTE" \
  --progress \
  --transfers 4 \
  --checkers 8
```

## 12. Quick reminders

- `mask_mode="waymo"` is not enough by itself.
- You must also use:
  - `--config=cosmos_transfer2/singleview_mask_config.py`
  - `data_train=example_singleview_train_data_depth_mask`
- Otherwise the training may still go through the non-mask dataset loader.
- If `image_context_mask_reference_path` is set, that PNG overrides the built-in mask shape.

## 13. Short version

If you only want the one command that matters most for a custom mask PNG, it is this:

```bash
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
  --config=cosmos_transfer2/singleview_mask_config.py \
  -- \
  data_train=example_singleview_train_data_depth_mask \
  experiment=transfer2_singleview_posttrain_pcd_rgb_image_context_example \
  dataloader_train.dataset.dataset_dir="/workspace/cosmos-transfer2.5-pcd/datasets/cosmos_train_2000" \
  'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
  dataloader_train.dataset.input_video_dir="pcd_videos" \
  dataloader_train.dataset.target_video_dir="rgb_videos" \
  dataloader_train.dataset.input_video_suffix="_vis_pcd" \
  dataloader_train.dataset.control_video_dir_override="pcd_videos" \
  dataloader_train.dataset.control_video_suffix="_vis_pcd" \
  dataloader_train.dataset.use_image_context=True \
  dataloader_train.dataset.image_context_from_rgb_first_frame=True \
  dataloader_train.dataset.mask_image_context=True \
  dataloader_train.dataset.image_context_mask_reference_path="/workspace/cosmos-transfer2.5-pcd/masks/mask_common_1024x512.png" \
  model.config.freeze_base_model=False \
  model.config.hint_keys=depth \
  model.config.fsdp_shard_size=8 \
  optimizer.lr=1e-4 \
  trainer.max_iter=2000 \
  checkpoint.save_iter=500
```
