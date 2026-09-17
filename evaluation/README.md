# Paper Evaluation Framework

`python -m evaluation.run_eval` is the only evaluation CLI. It uses a frozen
manifest, independent metric plugins, experiment YAML, and a shared
aggregation layer. It never downloads data, recursively discovers videos, or
replaces a missing official metric with a proxy.

## Layout

- `manifests/benchmark40.csv`: frozen 20 ID plus 20 NuScenes OOD benchmark.
- `configs/model_progression.yaml`: official Cosmos versus Stage 1 versus Stage 2.
- `configs/blend_ablation.yaml`: Stage 2 hard-lock versus blended decoder.
- `metrics/`: isolated metric implementations and explicit dependency metadata.
- `outputs/`: selected by each config or `--output-dir`; contains JSONL raw
  records, per-scene CSV, aggregate CSV, audit JSON, and Markdown summary.

## Required Paths

All paths are supplied by environment variables so the config remains portable:

```bash
export EVAL_DATA_ROOT="/workspace/controlled_checkpoint_ablation_40/data"
export EVAL_PREDICTIONS_ROOT="/workspace/controlled_checkpoint_ablation_40/predictions"
export EVAL_OUTPUT_ROOT="/workspace/controlled_checkpoint_ablation_40/evaluation_outputs"
```

The configured layout is direct and explicit:

```text
$EVAL_DATA_ROOT/test/{rgb_videos,mask}/<stem>[ _mask].mp4
$EVAL_DATA_ROOT/ood/{rgb_videos,mask}/<stem>[ _mask].mp4
$EVAL_PREDICTIONS_ROOT/<method>/<split>/hardlock/<cohort>/<named prediction>.mp4
```

## Commands

Validate all resolved paths and metric availability without decoding a video.
Missing required files are emitted as `skipped` raw rows and make this command
exit nonzero; unavailable external metric dependencies remain explicitly
`unavailable` but do not invalidate available metrics:

```bash
python -m evaluation.run_eval \
  --config evaluation/configs/model_progression.yaml \
  --metrics all \
  --dry-run \
  --overwrite
```

Run only the available, configured pixel/seam metrics. `--resume` reuses only
successful scene/plugin rows. It retries failed or unavailable rows.

```bash
python -m evaluation.run_eval \
  --config evaluation/configs/model_progression.yaml \
  --metrics psnr ssim lpips seam known_region_psnr known_region_ssim \
  --resume
```

The program records unavailable metric setup requirements in `audit.json` and
raw rows. FID/FVD are deliberately marked dataset-level and unavailable until
this repository pins an official implementation, feature weights, sampling
scheme, and ERP projection policy. The same rule applies to VBench, Q-Align,
and MEt3R.
