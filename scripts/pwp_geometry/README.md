# PwP Geometry Ablation（球面面積 loss × 接縫修正）

2×2 ablation：Stage 1、Stage 2 和推論都走這套獨立 pipeline。原本的訓練、推論檔案一行都沒改。

| 方法（輸出資料夾名稱） | `pwp_area_loss` | `pwp_vae_w_padding` | `pwp_sphere_pos_emb` |
|---|---|---|---|
| `baseline` | `off` | `zero` | `off` |
| `baseline_arealoss` | `sphere_area` | `zero` | `off` |
| `baseline_seam` | `off` | `circular` | `xyz` |
| `baseline_arealoss_seam` | `sphere_area` | `circular` | `xyz` |

- **sphere_area**：每個 ERP latent 列的平方誤差乘上 `cos(緯度)`（即該列的球面面積），再除以權重總和。上下 reflection padding 的列權重為 0。Stage 2 時權重是 `valid mask × cos(緯度)`。
- **circular**：Wan2.1 VAE 所有卷積只把「寬度方向」的 padding 改成左右環繞；高度維持 zero，時間維持 causal，VAE 權重不變。Stage 2 的 loss-mask guard 和推論 hardlock 的 erode 也會跟著環繞。
- **xyz**：DiT 主幹和 control 分支的 token 加上 `MLP(球面座標 xyz)`，輸出層零初始化（第 0 步和原模型完全相同）。原本的 3D RoPE 不變。

### 畫面尺寸 `pwp_frame_size`（四組共用，不屬於方法）

| 值 | 訓練 | 推論 |
|---|---|---|
| `1280x640`（這次 ablation 用這個） | 2:1 影片直接縮放成 1280×640，**沒有 padding**（`padding_mask` 全 0） | 2:1 輸入讀成 1280×640，**不拉伸** |
| `legacy_720`（舊行為，只用來重現舊結果） | 2:1 放進 16:9 桶：1280×640 + 上下各 32 px reflection padding = 1280×704 | 拉伸成 1280×704 |

- `1280x640` 是在這個 process 裡新增一個解析度 key `pwp_erp_1280x640`，原本的 `"720"` 等 key 完全不變，所以舊 pipeline 不受影響。
- 訓練時 `dataloader_train.dataset.resolution` 由 `pwp_frame_size` 自動決定，不用另外寫。
- `1280x640` 只接受 2:1 的輸入；推論時 control video 不是 2:1 會直接報錯，避免偷偷拉伸。
- 換算：1280×640 → latent 160×80 → DiT token 80×40（舊的是 80×44）。

## 設定一定要寫在指令上

三個開關和 `pwp_frame_size` **都沒有預設值**，漏寫任何一個就會在組 config 時報錯，並指出要加哪一行。
輸出資料夾名稱由開關自動決定（`legacy_720` 會在最後加 `_legacy720`）：

```
<IMAGINAIRE_OUTPUT_ROOT>/cosmos_transfer2_posttrain/pwp_geometry/stage1_baseline_arealoss_seam/
<IMAGINAIRE_OUTPUT_ROOT>/cosmos_transfer2_posttrain/pwp_geometry/stage2_baseline_arealoss_seam/
```

## 會擋下來的只有三種真正的錯誤

1. 開關或 `pwp_frame_size` 沒寫或寫錯值；`1280x640` 推論時輸入不是 2:1。
2. 載入的 checkpoint 是用不同方法訓練的。例如用 `stage1_baseline` 的 checkpoint 去跑 `baseline_seam` 的 Stage 2，或推論時的 `--pwp-*` 和 checkpoint 不一致。
   - 方法紀錄存在 checkpoint 權重裡（`net.pwp_method_stamp`），DCP、`.pt`、EMA 轉檔都會一起帶著，換機器也不會遺失。
   - 官方或舊 pipeline 的 checkpoint 沒有這筆紀錄，視為三個開關都關閉：baseline 可以接，seam 不行。
3. 開了 `circular` 又開了 VAE 的空間 context parallel（`--enable-parallel-tokenizer`），或 DiT 被切成空間 CP。

其他資訊都只記錄、不會擋：
- `pwp_frame_size` 不存進 checkpoint，也不會和 checkpoint 比對（推論時請自己用和訓練相同的值）；它會寫進下面兩種紀錄。
- 每次啟動（含接續訓練）會在輸出資料夾的 `pwp_run_records.jsonl` 加一行，內容包括方法、frame size、初始 checkpoint 的方法、git commit 和未 commit 的修改 hash、完整指令、load_path、dataset_dir。同樣的內容也會寫進 W&B config。
- 推論會在輸出資料夾寫 `pwp_inference_record.json`。

---

以下所有指令都在 repo 根目錄執行：`cd <REPO>/cosmos-transfer2.5-pcd && source .venv/bin/activate`。
`<...>` 是每台機器要自己填的路徑；**`<LR>` `<MAX_ITER>` `<SAVE_ITER>` 是待討論的訓練參數**。

## 0. 測試（CPU，幾秒）

```bash
python -m pytest tests/test_pwp_geometry.py -q
```

## 1. 確認 config 只差在預期的地方（選用，不會擋）

`--dryrun` 只會組出 config 並寫成 `config.yaml`，不會訓練：

```bash
IMAGINAIRE_OUTPUT_ROOT=/tmp/pwp_dryrun python -m scripts.train --dryrun \
  --config=cosmos_transfer2/pwp_geometry_config.py -- \
  experiment=pwp_geometry_stage1 \
  model.config.pwp_area_loss=off model.config.pwp_vae_w_padding=zero model.config.pwp_sphere_pos_emb=off \
  model.config.pwp_frame_size=1280x640

IMAGINAIRE_OUTPUT_ROOT=/tmp/pwp_dryrun python -m scripts.train --dryrun \
  --config=cosmos_transfer2/pwp_geometry_config.py -- \
  experiment=pwp_geometry_stage1 \
  model.config.pwp_area_loss=sphere_area model.config.pwp_vae_w_padding=circular model.config.pwp_sphere_pos_emb=xyz \
  model.config.pwp_frame_size=1280x640

python scripts/pwp_geometry/compare_configs.py \
  /tmp/pwp_dryrun/cosmos_transfer2_posttrain/pwp_geometry/stage1_baseline/config.yaml \
  /tmp/pwp_dryrun/cosmos_transfer2_posttrain/pwp_geometry/stage1_baseline_arealoss_seam/config.yaml
```

預期只會出現 `model.config.pwp_*`、`job.name` 和 `pwp_record_job_path` 的差異。兩份 `config.yaml` 的 `dataloader_train.dataset.resolution` 都應該是 `pwp_erp_1280x640`。

## 2. VAE 接縫測試（只推論，不用訓練）

用原本的 VAE 權重，把同一批 360° 影片分別以 zero 和 circular padding 做 encode→decode：

```bash
torchrun --nproc_per_node=1 scripts/pwp_geometry/vae_seam_test.py \
  --video-dir <360 RGB 影片資料夾> \
  --output-dir <OUTPUT>/vae_seam_test \
  --max-videos 20 \
  --frame-size 1280x640
```

輸出：
- `summary.json` 和 `per_video.csv`：
  - `seam_ratio`：接縫兩側的落差除以一般相鄰欄的落差，約 1 表示接縫和其他地方一樣平滑。
  - `psnr_edges`：左右邊緣 32 px 的重建 PSNR。
  - `psnr_interior`：其餘區域的重建 PSNR。
- `seam_crops/*.png`：把畫面平移半圈，讓接縫位在正中央的裁切圖。由上到下依序是原圖、zero、circular。

## 3. Stage 1（mask pool，從官方 depth checkpoint 開始）

四組只差三個方法開關，`pwp_frame_size` 和其餘設定完全相同。以下是 `baseline_arealoss_seam`：

```bash
export IMAGINAIRE_OUTPUT_ROOT=<OUTPUT_ROOT>
torchrun --nproc_per_node=8 --master_port=29501 -m scripts.train \
  --config=cosmos_transfer2/pwp_geometry_config.py -- \
  experiment=pwp_geometry_stage1 \
  model.config.pwp_area_loss=sphere_area \
  model.config.pwp_vae_w_padding=circular \
  model.config.pwp_sphere_pos_emb=xyz \
  model.config.pwp_frame_size=1280x640 \
  dataloader_train.dataset.dataset_dir=<STAGE1_DATA_ROOT> \
  'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
  dataloader_train.dataset.input_video_dir=pcd_videos \
  dataloader_train.dataset.target_video_dir=rgb_videos \
  dataloader_train.dataset.input_video_suffix=_pcd \
  dataloader_train.dataset.control_video_dir_override=pcd_videos \
  dataloader_train.dataset.control_video_suffix=_pcd \
  dataloader_train.dataset.use_image_context=True \
  dataloader_train.dataset.image_context_from_rgb_first_frame=True \
  dataloader_train.dataset.mask_image_context=True \
  dataloader_train.dataset.mask_depth_control=True \
  dataloader_train.dataset.mask_pool_root=<MASK_POOL_ROOT> \
  dataloader_train.dataset.mask_pool_use_dynamic=True \
  dataloader_train.dataset.mask_pool_sampler_py="$PWD/tools/mask_pool_sampler.py" \
  dataloader_train.dataset.mask_pool_sampler_kwargs_json="$PWD/scripts/pwp_geometry/mask_pool_sampler_config.json" \
  model.config.freeze_base_model=False \
  model.config.hint_keys=depth \
  model.config.min_num_conditional_frames=1 \
  model.config.max_num_conditional_frames=1 \
  model.config.fsdp_shard_size=8 \
  model_parallel.context_parallel_size=1 \
  optimizer.lr=<LR> \
  trainer.max_iter=<MAX_ITER> \
  checkpoint.save_iter=<SAVE_ITER> \
  '~trainer.callbacks.every_n_sample_reg' \
  '~trainer.callbacks.every_n_sample_ema' \
  checkpoint.load_path=<OFFICIAL_DEPTH_CKPT>.pt \
  checkpoint.load_training_state=False \
  checkpoint.strict_resume=False \
  job.wandb_mode=online \
  2>&1 | tee <OUTPUT_ROOT>/stage1_baseline_arealoss_seam.log
```

中斷後接續訓練：再執行一次完全相同的指令，會從輸出資料夾裡最新的 checkpoint 接續。

## 4. Stage 2（target loss mask，從「同一方法」的 Stage 1 checkpoint 開始）

```bash
export IMAGINAIRE_OUTPUT_ROOT=<OUTPUT_ROOT>
torchrun --nproc_per_node=8 --master_port=29501 -m scripts.train \
  --config=cosmos_transfer2/pwp_geometry_config.py -- \
  experiment=pwp_geometry_stage2 \
  model.config.pwp_area_loss=sphere_area \
  model.config.pwp_vae_w_padding=circular \
  model.config.pwp_sphere_pos_emb=xyz \
  model.config.pwp_frame_size=1280x640 \
  dataloader_train.dataset.dataset_dir=<STAGE2_DATA_ROOT> \
  'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
  dataloader_train.dataset.control_video_suffix=_pcd \
  model.config.fsdp_shard_size=8 \
  model_parallel.context_parallel_size=1 \
  optimizer.lr=2e-5 \
  'scheduler.cycle_lengths=[1120]' \
  trainer.max_iter=1120 \
  checkpoint.save_iter=280 \
  checkpoint.load_path=<OUTPUT_ROOT>/cosmos_transfer2_posttrain/pwp_geometry/stage1_baseline_arealoss_seam/checkpoints/iter_<NNNNNNNNN>/model_ema_bf16.pt \
  checkpoint.load_training_state=False \
  checkpoint.strict_resume=False \
  job.wandb_mode=online \
  2>&1 | tee <OUTPUT_ROOT>/stage2_baseline_arealoss_seam.log
```

- 資料是 Dropbox `driving_dataset/train`（2240 部），PCD 檔名是 `<stem>_pcd.mp4`；其餘資料夾（`rgb_videos/ mask/ prompts/`）用 experiment 預設值。
- LR、scheduler、iteration、`context_parallel_size=1` 和之前正式的 Stage 2（`training_stage2.md`）相同。
- 從 Stage 1 最後一個 checkpoint 的 EMA 權重 weights-only 開始（和之前 Stage 1 → Stage 2 的做法相同）。`.pt` 裡也帶著方法紀錄；在另一台機器上的話，只要把這個檔案複製過來。

## 5. 轉檔並推論（hardlock + no_video_path，erode 8 px）

```bash
python scripts/convert_distcp_to_pt.py \
  <OUTPUT_ROOT>/cosmos_transfer2_posttrain/pwp_geometry/stage2_baseline_arealoss_seam/checkpoints/iter_<NNNNNNNNN> \
  <OUTPUT_ROOT>/converted/stage2_baseline_arealoss_seam_iter_<NNNNNNNNN>

python run_cosmos_inference_batch.py <EVAL_DATA_ROOT> <INFER_OUTPUT>/stage2_baseline_arealoss_seam \
  --cosmos-root "$PWD" \
  --inference-script examples/inference_pwp_hardlock_no_video_path.py \
  --checkpoint <OUTPUT_ROOT>/converted/stage2_baseline_arealoss_seam_iter_<NNNNNNNNN>/model_ema_bf16.pt \
  --experiment pwp_geometry_inference_hardlock \
  --config-file cosmos_transfer2/pwp_geometry_config.py \
  --mode no-video-path \
  --guided-hardlock \
  --mask-erode-px 8 \
  --steps 35 --guidance 7 --seed 2025 \
  --inference-arg=--pwp-vae-w-padding=circular \
  --inference-arg=--pwp-sphere-pos-emb=xyz \
  --inference-arg=--pwp-frame-size=1280x640
```

推論只需要兩個方法開關，因為 area loss 只影響訓練。`baseline` 和 `baseline_arealoss` 用 `zero` / `off`。
`--pwp-frame-size` 會取代輸入 JSON 裡的 `resolution`；輸出仍會縮放回輸入影片的原始尺寸（例如 1024×512）。

## 檔案

| 檔案 | 用途 |
|---|---|
| `cosmos_transfer2/_src/transfer2/pwp_geometry/method.py` | 開關、checkpoint 方法紀錄、輸出名稱、git 紀錄 |
| `.../pwp_geometry/frame_size.py` | `1280x640` / `legacy_720` → 解析度 key、2:1 檢查 |
| `.../pwp_geometry/area_weight.py` | cos(緯度) 列權重 |
| `.../pwp_geometry/vae_circular.py` | 把已載入的 Wan VAE 寬度 padding 改成環繞 |
| `.../pwp_geometry/masks.py` | 環繞版 Stage 2 guard 和推論 erode |
| `.../pwp_geometry/sphere_pos_emb.py` | token 經緯度 → xyz → MLP（零初始化） |
| `.../pwp_geometry/network.py` | DiT 子類別：XYZ embedding + 方法紀錄 buffer |
| `.../pwp_geometry/model.py` | 訓練 model：loss 權重、VAE、方法檢查、run record |
| `.../pwp_geometry/inference_model.py`、`inference_pipeline.py` | 推論 model 和 pipeline |
| `cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/defaults/model_pwp_geometry.py` | 註冊 model |
| `cosmos_transfer2/experiments/singleview/pwp_geometry.py` | `pwp_geometry_stage1` / `_stage2` / `_inference_hardlock` |
| `cosmos_transfer2/pwp_geometry_config.py` | `--config` 入口 |
| `cosmos_transfer2/inference_pwp_hardlock_no_video_path.py`、`examples/inference_pwp_hardlock_no_video_path.py` | 推論入口 |
| `scripts/pwp_geometry/vae_seam_test.py`、`compare_configs.py`、`mask_pool_sampler_config.json` | 工具 |
| `tests/test_pwp_geometry.py` | CPU 測試 |

## 注意

- W&B 的 `train/loss` 在開了 area loss 時是加權後的值，不能直接和 baseline 的曲線比較。
- `1280x640` 讓訓練和推論都是同一個 1280×640、無 padding 的畫面，修掉了舊流程「訓練補 padding、推論拉伸」的不一致。
- 預訓練沒看過「沒有 padding 的 80×40 token」這個組合（RoPE 位置都在看過的範圍內，權重不變），建議先比一組 `baseline` 的 `1280x640` 和 `legacy_720`，確認沒有變差再跑完四組。
- 舊 checkpoint（用 `legacy_720` 訓練的）推論時請用 `--pwp-frame-size legacy_720`，才會和它們原本的數字一致。
