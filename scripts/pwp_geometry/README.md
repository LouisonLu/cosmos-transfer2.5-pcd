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

## 3. Stage 1（any-view mask，從官方 depth checkpoint 開始）

### Any-view mask（`pwp_geometry/anyview_masks.py`，四組相同）

每個樣本抽一張 mask（白 = 看得到），同時套在首幀和 PCD。比例和遮擋設定在 `scripts/pwp_geometry/anyview_mask_sampler_config.json`：

| 類別 | 比例 | 對應的真實輸入 |
|---|---|---|
| `driving_real` | 10% | 真實車載相機 mask（AV2 / Waymo / nuScenes / PandaSet），讀 `<mask_pool>/train/driving_real/` 的 200 張 |
| `driving_front` | 15% | 單一車頭相機（single front-view 評估） |
| `driving_multi` | 15% | 3–7 顆車用相機，隨機少 0–3 顆 |
| `single_camera` | 25% | 任何方向的單一矩形相機：手機、手持、機器人頭部、俯拍、仰拍；橫幅、直幅、正方 |
| `fisheye` | 5% | 150–220° 魚眼，有時朝正上或正下 |
| `multi_camera` | 10% | 機器人頭 + 胸、無人機朝下、立體相機、任意多相機 |
| `partial_pano` | 5% | 手機全景（90–270° 寬的帶） |
| `pano_holes` | 2% | 有缺洞的 360：腳架、天頂、雙魚眼接縫（±90°）、遮擋物 |
| `near_black` | 5% | 幾乎全黑：極窄的相機或少數散點 |
| `no_mask` | 8% | 完整首幀（driving50 評估） |

- 視野內物體遮擋：上表前五類有 40% 會再遮掉 1–6 個物體形狀（車輛方塊、橢圓、不規則多邊形，多在下半部），合計約 5–30% 的可見像素。
- 除了 `driving_real` 的檔案，其他都在訓練時產生。每個 (epoch, 影片) 的 mask 只由 `(mask_pool_sampler_seed, epoch, 影片 index)` 決定，所以可以在訓練前匯出全部、逐張檢查（和訓練時逐像素相同，見下方指令與 CPU 測試）：

```bash
python scripts/pwp_geometry/export_anyview_masks.py training \
  --mask-pool-root <MASK_POOL_ROOT> --dataset-dir <STAGE1_DATA_ROOT> --epochs 2 --out <OUT>/stage1_training_masks
# 固定的 val mask（評估沒看過的視角用）
python scripts/pwp_geometry/export_anyview_masks.py val --mask-pool-root <MASK_POOL_ROOT> --count 500 --seed 1 --out <OUT>/val_masks
```

`<MASK_POOL_ROOT>` 是 Dropbox 的 `360 video/mask_pool`（`train/driving_real` 200 張、`val/driving_real` 50 張，由 `select_driving_real_masks.py` 從 `mask_pool(old)` 挑出版型最不同的）。

### 訓練

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
  dataloader_train.dataset.mask_pool_sampler_py="$PWD/cosmos_transfer2/_src/transfer2/pwp_geometry/anyview_masks.py" \
  dataloader_train.dataset.mask_pool_sampler_kwargs_json="$PWD/scripts/pwp_geometry/anyview_mask_sampler_config.json" \
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

中斷後接續訓練：同一段指令，只把最後兩個 checkpoint 參數換成要接續的 checkpoint 資料夾（含 `model/`、`optim/`），用官方的完整接續（模型含 EMA、optimizer、LR scheduler、iteration）：

```bash
  checkpoint.load_path=<OUTPUT_ROOT>/cosmos_transfer2_posttrain/pwp_geometry/stage1_<GROUP>/checkpoints/iter_<NNNNNNNNN> \
  checkpoint.load_training_state=True \
```

sampler 從這個 iteration 算出 epoch 和位置接著讀（log 會印 `PwP sampler: ... start at iteration N: epoch E, skipping K videos`），接續後讀到的影片順序和 mask 和沒中斷時相同。換機器時把整個 `iter_<NNNNNNNNN>` 資料夾複製過去，`load_path` 指到新位置即可。

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

基礎版就是 hardlock + no_video_path：只給第一幀（mask 外為黑），後面的幀全黑，避免整段 RGB 洩漏資訊。下面三個推論技巧預設全部關閉，要在指令上加旗標才會用。

```bash
python scripts/convert_distcp_to_pt.py \
  <OUTPUT_ROOT>/cosmos_transfer2_posttrain/pwp_geometry/stage1_baseline_arealoss_seam/checkpoints/iter_<NNNNNNNNN> \
  <OUTPUT_ROOT>/converted/stage1_baseline_arealoss_seam_iter_<NNNNNNNNN>

python run_cosmos_inference_batch.py <EVAL_DATA_ROOT> <INFER_OUTPUT>/stage1_baseline_arealoss_seam \
  --cosmos-root "$PWD" \
  --inference-script examples/inference_pwp_hardlock_no_video_path.py \
  --checkpoint <OUTPUT_ROOT>/converted/stage1_baseline_arealoss_seam_iter_<NNNNNNNNN>/model_ema_bf16.pt \
  --experiment pwp_geometry_inference_hardlock \
  --config-file cosmos_transfer2/pwp_geometry_config.py \
  --mode no-video-path \
  --guided-hardlock \
  --mask-erode-px 8 \
  --steps 35 --guidance 7 --seed 2025 \
  --inference-arg=--pwp-vae-w-padding=circular \
  --inference-arg=--pwp-sphere-pos-emb=xyz
```

推論只需要兩個方法開關（會和 checkpoint 的方法紀錄比對），因為 area loss 只影響訓練：

| checkpoint | `--pwp-vae-w-padding` | `--pwp-sphere-pos-emb` |
|---|---|---|
| `baseline`、`baseline_arealoss` | `zero` | `off` |
| `baseline_seam`、`baseline_arealoss_seam` | `circular` | `xyz` |

畫面尺寸預設 `1280x640`（和訓練相同），會取代輸入 JSON 裡的 `resolution`：2:1 的輸入（例如 1024×512）等比例縮放到 1280×640 生成，輸出再縮回輸入的原始尺寸；不是 2:1 的輸入會直接報錯。`legacy_720` 是舊流程（訓練：1280×640 上下各補 32 px 反射 padding 成 1280×704；推論：直接拉伸成 1280×704），只有用舊流程訓練的 checkpoint 才需要 `--inference-arg=--pwp-frame-size=legacy_720`。

### 選用功能（預設關閉）

| 旗標 | 作用 |
|---|---|
| `--inference-arg=--pwp-mask-pcd` | PCD control 每一幀都用第 0 幀的 hardlock mask（不 erode）遮掉，mask 外為黑，和 Stage 1 訓練的 `mask_depth_control=True` 相同 |
| `--inference-arg=--pwp-blended-decoding` | Dual-orientation blended decoding：最後的 latent 以 0° 和 180° 各解碼一次再混合，接縫附近用轉過的那次 |
| `--inference-arg=--pwp-latent-rotation` | Latent rotation during denoising：每一步把 latent 連同所有對齊的條件（hardlock 的 GT 和 mask、control、control weight、padding mask、XYZ 位置編碼）水平平移 patch 寬度的隨機倍數，網路跑完再平移回來；sampler 永遠在原方向。平移量由 seed 決定，會寫進 log |
| `--inference-arg=--pwp-latitude-noise-strength=<S>` | Latitude-aware noise：緯度 φ 的那一列初始 noise 只有 `round(W cos φ)` 個獨立值、沿圓周展開，每個值仍是 N(0,1)；以 `sqrt(1-S)·iid + sqrt(S)·緯度 noise` 混合，`S=0` 就是原本的 noise。模型訓練時沒看過這種 noise |

- 全部關閉時直接呼叫原本的取樣迴圈，結果和原本的 hardlock 一樣；開了技巧也只包住初始 noise 和網路呼叫，迴圈本身不變。
- Latent rotation 和 latitude noise 不能和 context parallel 一起用（需要整個 latent 在同一張 GPU）。
- 開了哪些功能會寫進輸出資料夾的 `pwp_inference_record.json`。不同設定請用不同的輸出資料夾。

### 檢查 mask 邊界的黑色會不會被鎖進去（只輸出報告，不生成）

hardlock 鎖的是「遮掉的第一幀」的 latent。VAE 編碼時會看到可見區外的黑色，所以 mask 邊界附近的 latent 可能和真實畫面的 latent 不同。這個工具用和推論完全相同的讀圖、mask erosion、VAE（zero 或 circular）和 latent mask，比較遮掉的第一幀和完整第一幀的 latent，列出每個 `--erode-px` 下被鎖住的 latent 格的誤差、依「離 mask 邊界幾 px」的誤差，以及 erosion 後 mask 內接近全黑的像素比例。

```bash
python run_cosmos_inference_batch.py <EVAL_DATA_ROOT> <CHECK_OUTPUT>/prepared \
  --cosmos-root "$PWD" \
  --inference-script examples/inference_pwp_hardlock_no_video_path.py \
  --checkpoint <CKPT>.pt \
  --experiment pwp_geometry_inference_hardlock \
  --config-file cosmos_transfer2/pwp_geometry_config.py \
  --mode no-video-path --guided-hardlock --mask-erode-px 8 \
  --prepare-only

python scripts/pwp_geometry/check_hardlock_black_leak.py \
  --work-root <CHECK_OUTPUT>/prepared/_work -o <CHECK_OUTPUT>/report \
  --checkpoint-path <CKPT>.pt \
  --experiment pwp_geometry_inference_hardlock \
  --config-file cosmos_transfer2/pwp_geometry_config.py \
  --disable-guardrails \
  --pwp-vae-w-padding zero --pwp-sphere-pos-emb off \
  --erode-px 0 8 16 24 32 48
```

誤差 = 兩個 latent 在每格的 channel L2 距離 ÷ 完整畫面 latent 的 channel 範數中位數。離邊界很遠的格子是底線（應接近 0）。每個樣本另外輸出一張誤差圖（白線是 mask 邊界）。

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
| `.../pwp_geometry/data.py` | 每個 epoch 重洗的 sampler、依 (epoch, 影片) 抽 mask 的 Stage 1 dataset |
| `.../pwp_geometry/anyview_masks.py` | any-view mask：相機產生器、物體遮擋、`MaskPoolSampler`（`mask_pool_sampler_py`） |
| `.../pwp_geometry/inference_model.py`、`inference_pipeline.py` | 推論 model 和 pipeline |
| `.../pwp_geometry/inference_techniques.py` | 推論選用功能：PCD mask、latent rotation、latitude-aware noise（blended decoding 沿用原本的函式） |
| `cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/defaults/model_pwp_geometry.py` | 註冊 model |
| `cosmos_transfer2/experiments/singleview/pwp_geometry.py` | `pwp_geometry_stage1` / `_stage2` / `_inference_hardlock` |
| `cosmos_transfer2/pwp_geometry_config.py` | `--config` 入口 |
| `cosmos_transfer2/inference_pwp_hardlock_no_video_path.py`、`examples/inference_pwp_hardlock_no_video_path.py` | 推論入口 |
| `scripts/pwp_geometry/anyview_mask_sampler_config.json` | any-view mask 的比例與遮擋設定 |
| `scripts/pwp_geometry/export_anyview_masks.py`、`select_driving_real_masks.py` | 匯出訓練 / val mask；挑選 driving_real |
| `scripts/pwp_geometry/vae_seam_test.py`、`compare_configs.py`、`check_hardlock_black_leak.py` | 工具 |
| `tests/test_pwp_geometry.py` | CPU 測試 |

## 注意

- 資料抽樣（`pwp_geometry/data.py`，四組相同）：每個 epoch 重新洗牌影片順序；Stage 1 的 mask 由 `(mask_pool_sampler_seed, epoch, 影片 index)` 決定，所以每張 GPU、每個 worker、每個 epoch 抽到的都不同，同一組 (epoch, 影片) 每次都抽到同一張。原本的流程沒有 `set_epoch`，且 mask sampler 只用 `seed + worker_id`，8 張 GPU 每步同一張 mask、每個 epoch 重複同一串 mask。用上面的接續指令（`load_path=.../iter_<N>` + `load_training_state=True`）接續時，sampler 從第 N 個 iteration 的 epoch 和位置接著讀；其他情況（例如從官方 `.pt` 開始）一律從第 0 個 epoch 開始。
- W&B 的 `train/loss` 在開了 area loss 時是加權後的值，不能直接和 baseline 的曲線比較。
- `1280x640` 讓訓練和推論都是同一個 1280×640、無 padding 的畫面，修掉了舊流程「訓練補 padding、推論拉伸」的不一致。
- 預訓練沒看過「沒有 padding 的 80×40 token」這個組合（RoPE 位置都在看過的範圍內，權重不變），建議先比一組 `baseline` 的 `1280x640` 和 `legacy_720`，確認沒有變差再跑完四組。
- 舊 checkpoint（用 `legacy_720` 訓練的）推論時請用 `--pwp-frame-size legacy_720`，才會和它們原本的數字一致。
- Stage 1 訓練時第 0 幀的條件是 masked PCD（整張鎖定），hardlock 推論則是 masked RGB、只鎖已知區。四組的差異相同，所以彼此仍可比較；Stage 2 訓練的第 0 幀是 masked RGB，和推論較一致。
