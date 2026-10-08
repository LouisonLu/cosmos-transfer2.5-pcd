# Cosmos Transfer2.5: PCD and Masked-Frame Extensions

This repository contains research extensions around NVIDIA Cosmos Transfer2.5 for
PCD/depth-conditioned 360-video generation, masked first-frame training, partial
hardlock inference, and controlled evaluation.

The upstream Cosmos implementation remains the foundation. The additions in this
repository are intentionally organized around the following maintained paths:

- PCD video used as the depth control input.
- Optional masked first-frame image context.
- Partial hardlock inference with an explicit guided-generation mask.
- Target-loss-mask training and dataset validation utilities.
- Bounded batch inference with resumable progress reporting.

## Public release boundary

Datasets, checkpoints, generated videos, training outputs, credentials, and local
environment files are not part of the public code release. Store them outside the
repository and pass their locations through command-line arguments or environment
variables.

Never put a Hugging Face, GitHub, W&B, Dropbox, or cloud token in a JSON file, shell
script, command-line URL, or committed environment file. Use the provider's login
command or an environment variable managed outside the repository.

## Requirements

- Linux x86-64
- NVIDIA GPU with Ampere architecture or newer
- A recent NVIDIA driver compatible with the selected CUDA extra
- `git`, `git-lfs`, `ffmpeg`, and [`uv`](https://docs.astral.sh/uv/)
- Enough storage for the model cache, dataset, and generated outputs

The exact GPU memory requirement depends on the model, resolution, frame count, and
parallelism. Start with a one-sample dry run before launching a large job.

## Installation

```bash
git clone https://github.com/LouisonLu/cosmos-transfer2.5-pcd.git
cd cosmos-transfer2.5-pcd

git lfs install
uv sync --extra=cu128
source .venv/bin/activate
```

Use `--extra=cu130` when the host requires the CUDA 13 build. Do not install private
dependencies by copying a local virtual environment into the repository.

Authenticate outside project files when a model or dataset requires it:

```bash
uv run --with huggingface-hub hf auth login
wandb login
```

Check the environment:

```bash
python scripts/check_environment.py
nvidia-smi
```

## Dataset layout

For the masked PCD training path, use a dataset root outside this checkout:

```text
<dataset-root>/
├── rgb_videos/
│   └── <stem>.mp4
├── pcd_videos/
│   └── <stem>_pcd.mp4
└── captions/
    └── <stem>.json
```

Each caption JSON must contain a non-empty `caption`, `text`, or `prompt` field.
RGB, PCD, and caption stems must match. A custom first-frame mask can be supplied
as a separate local PNG or as a mask video for inference.

Keep dataset and output roots outside the checkout, for example:

```bash
export REPO_ROOT="$(pwd)"
export DATASET_ROOT="/path/to/pcd-dataset"
export OUTPUT_ROOT="/path/to/cosmos-output"
export CHECKPOINT="/path/to/model_ema_bf16.pt"
```

## Masked PCD training

The maintained training recipe is documented in
[`train_firstframe_mask.md`](train_firstframe_mask.md). It trains with:

```text
input  = masked first RGB frame + PCD video + prompt
target = full RGB 360 video
```

The important switches are the masked dataset config, the `_pcd` file suffix, and
`mask_image_context=True`. The training command writes checkpoints under the
external output root supplied by the user.

Before a full run, validate one or a few samples and use a small `trainer.max_iter`.

## Partial hardlock inference

The batch runner expects this layout when masks are required:

```text
<inference-data>/
├── rgb_videos/<stem>.mp4
├── pcd_videos/<stem>_pcd.mp4
├── mask/<stem>_mask.mp4
└── prompts/<stem>_prompt.json
```

Run a bounded dry run first:

```bash
python run_cosmos_inference_batch.py \
  "$DATASET_ROOT" "$OUTPUT_ROOT" \
  --cosmos-root "$REPO_ROOT" \
  --inference-script examples/inference_partial_hardlock_no_video_path.py \
  --checkpoint "$CHECKPOINT" \
  --experiment transfer2_singleview_partial_hardlock_pcd_rgb_image_context_example \
  --config-file cosmos_transfer2/singleview_partial_hardlock_config.py \
  --mode no-video-path \
  --dry-run
```

For a real one-sample run, remove `--dry-run` and add `--limit 1`. Use
`--prepare-only` when you want to inspect the generated request JSON without
starting Cosmos. Use `--resume` to skip outputs that already exist.

The standard-video path is also available:

```bash
python run_cosmos_inference_batch.py --help
```

This command lists all options without requiring a checkpoint or dataset.

## Validation and evaluation

Useful checks are kept small and explicit:

```bash
python tools/check_video_resolution.py --help
python tools/check_target_loss_mask_dataset.py --help
python tools/audit_controlled_checkpoint_compatibility.py --help
pytest -q tests/test_target_loss_mask.py
```

Evaluation reports and generated media belong under an external output root and
should not be committed.

## Repository map

```text
cosmos_transfer2/       model configs, inference entry points, and pipelines
examples/                user-facing inference entry points
scripts/                 setup, training, conversion, and data preparation tools
tools/                   bounded validators and research utilities
docs/                    upstream Cosmos documentation
train_firstframe_mask.md PCD + masked first-frame training guide
```

The `partial_hardlock` and `target_loss_mask` names identify maintained research
paths. Older spatial-lock, visible-preserve, and hardlock-off experiments are not
part of the current public entry points.

## Licensing

The source code is released under the Apache License 2.0. Cosmos model checkpoints
remain subject to the applicable NVIDIA model license. Review third-party licenses
before redistribution.
