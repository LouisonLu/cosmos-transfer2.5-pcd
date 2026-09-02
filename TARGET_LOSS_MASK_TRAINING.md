# Target-validity masked RGB training

This is an isolated post-training path. It does not change the existing mask-pool dataset, base rectified-flow model, or partial-hardlock inference.

## Input contract

Each split must use one canonical stem for all four inputs:

```text
<dataset_split>/
├── rgb_videos/
│   └── <stem>.mp4
├── pcd_videos/
│   └── <stem>_erp_pose.mp4
├── prompts/
│   └── <stem>_prompt.json
└── mask/
    └── <stem>_mask.mp4
```

- RGB is the masked target video `Y`.
- PCD remains full and is used as both the depth control and conditional video input.
- The first frame of masked RGB is the masked `image_context`.
- Prompt JSON must contain a non-empty `caption`, `text`, or `prompt` string.
- Mask video must contain at least the same 93 selected frames as RGB and PCD.
- White (`>=128`) means valid target information and contributes to loss.
- Black (`<128`) means an invalid stitched hole and contributes zero target loss.

The loader filters incomplete pairs at startup and logs the skipped stems rather than aborting training. It logs RGB/PCD mask inconsistencies as diagnostics. Source PCD may contain additional geometry in invalid regions because it is intentionally kept full as the depth control and conditional video input; only `Y`, its first-frame context, and the loss validity mask use the RGB mask.

## Validate local data

Check a fast first/middle/last-frame sample across the complete split:

```bash
python tools/check_target_loss_mask_dataset.py /data/driving_dataset/train \
  --pcd-suffix _erp_pose \
  --report /output/train_target_loss_mask_report.json
```

For the strictest check, decode all 93 frames:

```bash
python tools/check_target_loss_mask_dataset.py /data/driving_dataset/train \
  --pcd-suffix _erp_pose \
  --frame-indices all \
  --report /output/train_target_loss_mask_report_all_frames.json
```

Review RGB target warnings rather than ignoring them: they can indicate an inverted mask or incorrectly paired files. Use `--strict-pcd-mask` only when the source PCD itself is expected to be pre-masked.

## Training entry point

Use the new config and experiment only:

```bash
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
  --config=cosmos_transfer2/singleview_target_loss_mask_config.py \
  -- \
  experiment=transfer2_singleview_posttrain_pcd_masked_rgb_target_loss_mask \
  job.name=driving_target_loss_mask \
  dataloader_train.dataset.dataset_dir=/data/driving_dataset/train \
  'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
  dataloader_train.dataset.control_video_suffix=_erp_pose \
  model.config.fsdp_shard_size=8 \
  model_parallel.context_parallel_size=8 \
  trainer.max_iter=3000 \
  checkpoint.save_iter=250
```

Add the appropriate `checkpoint.load_path`, `checkpoint.load_training_state`, and optimizer settings for the intended fresh or resumed run.

## Loss behavior

The model computes the ordinary rectified-flow velocity target, but MSE is normalized only over valid latent elements:

```text
per_sample_loss = sum(valid_latent_mask * (predicted_velocity - target_velocity)^2)
                  / max(sum(valid_latent_mask), 1)
```

Before latent downsampling, invalid RGB-mask regions are dilated by the configurable spatial and temporal guards. Any latent cell overlapping an invalid region is excluded. Padding is always excluded. Context-parallel ranks normalize using the global valid-element count.

Default safety parameters:

```text
model.config.target_loss_mask_spatial_guard_px=8
model.config.target_loss_mask_temporal_guard_frames=1
model.config.target_loss_mask_min_valid_ratio=0.01
```

The guard reduces boundary contamination but also discards some valid supervision. Compare guard sizes on a short run before committing to full post-training.

## Important limitations

- Zero loss in invalid regions does not provide ground-truth supervision there. Completion quality still comes from the pretrained model, full PCD, masked first frame, and prompt.
- VAE features have a receptive field larger than a literal one-to-one pixel bin. The guard is conservative but cannot mathematically remove every boundary influence.
- Per-sample normalization gives sparse and dense masks equal sample weight. Samples with very little valid content should be filtered rather than allowed to dominate through a tiny denominator.
- Loss masks stored in lossy H.264 can acquire gray boundary pixels. Thresholding handles small artifacts, but lossless binary masks are preferable when storage permits.
- Poor or static PCD remains a strong condition and can degrade output even when the target loss mask is correct.
