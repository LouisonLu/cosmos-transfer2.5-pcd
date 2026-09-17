# Controlled Checkpoint Compatibility Audit

## Scope and provenance

- Audit type: static source/configuration audit only. No checkpoint was downloaded, deserialized, or executed.
- Repository: `cosmos-transfer2.5-pcd` at commit `f797ef3c8223f8c2bd545d4b71a326093e9403b7`.
- Intended entry point: `examples/inference_partial_hardlock_no_video_path.py`.
- Intended inference model config: `cosmos_transfer2/singleview_partial_hardlock_config.py`.
- Intended experiment: `transfer2_singleview_partial_hardlock_pcd_rgb_image_context_example`.
- Deliberately excluded: blended decoding, 40-scene batch execution, checkpoint download, and dry-run generation.

## Checkpoint inventory

| Label | Exact artifact | Training model/config evidence | Static loading conclusion |
| --- | --- | --- | --- |
| Official depth baseline | `nvidia/Cosmos-Transfer2.5-2B/general/depth/626e6618-bfcd-4d9a-a077-1409e2ce353f_ema_bf16.pt` | The source maps `ModelVariant.DEPTH` to UUID `626e6618-bfcd-4d9a-a077-1409e2ce353f` in `cosmos_transfer2/config.py:158-164`. Its public artifact is under `general/depth/` in the NVIDIA repository. | Candidate only. The official checkpoint was trained for depth control, but it was not statically proven to have an identical state dict to the custom partial-hardlock model. |
| Stage 1 epoch 08 | `LouisonLu/cosmos-transfer2.5-pcd-maskpool-pcdmasked-resume-common-mask-2000videos-epoch4/epoch_08/model_ema_bf16.pt` | Stage 1’s documented mask-pool training uses `cosmos_transfer2/singleview_mask_config.py` and `transfer2_singleview_posttrain_pcd_rgb_image_context_example`; its final cumulative checkpoint contains `epoch_08/model_ema_bf16.pt` (`train_mask_pool.md:389-423`, `:510-529`). The registered training class is `ControlVideo2WorldModelRectifiedFlow`. | Candidate only. The training class and the inference class differ, but the inference class is a separate model subclass intended to retain the same control architecture plus runtime mask handling. Actual key/shape parity remains unverified. |
| Stage 2 epoch 04 | `LouisonLu/cosmos-transfer2.5-pcd-driving2240videos-Ymasked-warmstart-pcdfixed-epoch4/epoch_04/model_ema_bf16.pt` | Stage 2 uses `cosmos_transfer2/singleview_target_loss_mask_config.py` and `transfer2_singleview_posttrain_pcd_masked_rgb_target_loss_mask`. Its training class is `ControlVideo2WorldModelTargetLossMaskRectifiedFlow` (`cosmos_singleview_target_loss_mask.py:16-74`; `model_target_loss_mask.py:14-27`). | Candidate only. This target-loss subclass changes training `forward`, not the requested inference entry point. Actual state-dict parity with partial-hardlock remains unverified. |

## Shared resolved inference contract

All three checkpoints must be invoked with the same entry point, model config, experiment name, and request fields below. The requested actual source paths were not supplied to this static audit, so they are intentionally represented by explicit placeholders rather than guessed paths.

```json
{
  "name": "<same_scene_and_run_name_per_checkpoint>",
  "prompt": "<same_prompt_text_from_the_same_prompt_json>",
  "seed": 2025,
  "guidance": 7,
  "image_context_path": "<same_masked_broken_first_rgb_frame.png>",
  "resolution": "720",
  "max_frames": 93,
  "num_conditional_frames": 1,
  "num_video_frames_per_chunk": 93,
  "num_steps": 35,
  "keep_input_resolution": true,
  "depth": {
    "control_path": "<same_corrected_complete_pcd_video.mp4>",
    "control_weight": 1.0,
    "mask_path": null,
    "mask_prompt": null
  },
  "guided_generation_mask": "<same_93_frame_mask_video.mp4>",
  "guided_generation_mask_first_frame_only": true,
  "guided_generation_mask_erode_px": 8,
  "guided_generation_step_threshold": 25
}
```

The entry reads a request JSON or JSONL object through `InferenceArgumentsNoVideoPath.from_files` (`examples/inference_partial_hardlock_no_video_path.py:65-81`; `cosmos_transfer2/config.py:353-391`). A raw prompt JSON that only contains `caption`, `text`, or `prompt` is **not** itself a complete request; it must first be converted to the schema above.

## Source-proven input behavior

| Requirement | Source proof | Controlled interpretation |
| --- | --- | --- |
| Image context | `InferenceArgumentsNoVideoPath.image_context_path` is required and documented as the only RGB appearance input (`config_no_video_path.py:40-43`). | Use the same masked/broken RGB frame 0 PNG for all models. |
| No full target RGB video | The no-video-path pipeline makes a surrogate video: image context at frame 0 and black frames afterward (`inference_pipeline_partial_hardlock_no_video_path.py:43-50`, `:90-123`). | A full target RGB video is not an inference input and must not be supplied as `video_path`. |
| PCD/depth control | `depth.control_path` is required in no-video-path mode (`config_no_video_path.py:104-127`). Precomputed depth is loaded through `read_and_process_control_input` (`inference_pipeline_partial_hardlock_no_video_path.py:204-212`; `utils.py:625-701`). | Use the same complete corrected PCD MP4 as the sole `depth` control, with weight `1.0`. |
| Supported control modalities | The entry supports `edge`, `depth`, `vis`, and `seg` (`examples/inference_partial_hardlock_no_video_path.py:38-46`; `config.py:477`). | The controlled experiment must activate only `depth`; adding a modality makes it a different inference condition. |
| Prompt | The request requires `prompt` (`config_no_video_path.py:79-83`). The default negative prompt is source-defined in `config.py:209-210`. | Keep both positive prompt and the source default negative prompt identical. Do not add a checkpoint-specific prompt. |
| Mask video | A guided mask may be MP4 or NPZ; MP4 is decoded as BCTHW and must have three channels before latent-map construction (`inference_pipeline_partial_hardlock.py:308-366`, `:368-407`). | Use the same 93-frame, three-channel mask MP4. It is a hard-lock mask, not a `depth.mask_path`. |
| Erosion | White regions are eroded at input resolution before latent hard-lock; image context is deliberately unchanged (`inference_pipeline_partial_hardlock_no_video_path.py:55-70`, `:162-182`). | Set `guided_generation_mask_erode_px=8` for every run. White means preserved/locked; black remains generative. |
| Conditional frames | The request value is `1`; the first single chunk is explicitly assigned one conditional frame (`config_no_video_path.py:47-48`; `inference_pipeline_partial_hardlock_no_video_path.py:295-300`). | With exactly 93 frames and a 93-frame chunk, there is one chunk and one conditional frame for every model. |
| Decoder | Every non-blended run calls `self.model.decode(sample)` after `generate_samples_from_batch` (`inference_pipeline_partial_hardlock_no_video_path.py:308-318`). | The decoder route is identical as long as the same entry point is used and the blended entry point is not used. |

## Partial hard-lock mathematics

`singleview_partial_hardlock_config.py:17-21` registers `fsdp_control_vace_rectified_flow_partial_hardlock`, whose instantiated class is `ControlVideo2WorldModelRectifiedFlowPartialHardlock` (`defaults/model_partial_hardlock.py:15-35`). The runtime path:

1. Converts the guided mask into a latent weight map (`inference_pipeline_partial_hardlock.py:368-407`).
2. Places it in `data_batch["partial_hardlock_mask"]` (`inference_pipeline_partial_hardlock_no_video_path.py:251-261`, `:295-300`).
3. Multiplies the normal conditional-frame mask by that spatial mask in both conditional and unconditional branches (`vid2vid_model_control_vace_rectified_flow_partial_hardlock.py:287-300`, `:352-369`).

This logic is checkpoint-independent. It is applied identically to all three requested runs **only if** all use this exact entry point, config file, experiment, and request fields.

## Architecture/config comparison

| Item | Official depth | Stage 1 epoch 08 | Stage 2 epoch 04 |
| --- | --- | --- | --- |
| Weight provenance | NVIDIA depth-control checkpoint | Mask-pool continuation of depth-control post-training | Warm-start from Stage 1 lineage with corrected PCD and target-validity masked loss |
| Training model class | Official source definition not available in this checkout as a standalone training run record | `ControlVideo2WorldModelRectifiedFlow` via default `fsdp_control_vace_rectified_flow` (`defaults/model.py:51-72`) | `ControlVideo2WorldModelTargetLossMaskRectifiedFlow` (`defaults/model_target_loss_mask.py:14-27`) |
| Training-specific change | None in this repo’s custom code | Masked image context and depth control are applied by the Stage 1 dataloader (`train_mask_pool.md:397-408`, `:531-548`) | `forward` computes the masked target loss and requires `target_loss_mask` only during training (`vid2vid_model_control_vace_rectified_flow_target_loss_mask.py:53-120`) |
| Inference model class | Same forced class for controlled comparison: `ControlVideo2WorldModelRectifiedFlowPartialHardlock` | Same | Same |
| Hint/control architecture requested at inference | `hint_keys="depth"` | `hint_keys="depth"` | `hint_keys="depth"` |
| Image-context architecture requested at inference | `use_reference_image=True`, `extra_image_context_dim=1152` | Same | Same |
| Hard-lock/erosion/decoder code path | Same | Same | Same |
| Checkpoint-specific inference branch | None in source, provided non-distilled checkpoints are used | None | None |

## What is feasible now

### Source-level result

The 40 preselected inputs can be used for an official native depth baseline **in the same no-video-path hard-lock protocol**, provided each scene has all four aligned artifacts:

1. masked/broken frame-0 RGB PNG;
2. complete corrected PCD MP4 with 93 frames;
3. three-channel mask MP4 with 93 frames;
4. request JSON containing the exact prompt and parameters above.

For a fair model-stage comparison, use exactly the same request object except for the output `name` and `--checkpoint-path`. The original RGB video must not be introduced for only the official baseline, because that changes the inference condition.

### Non-negotiable runtime preflight still required

This static audit cannot determine missing keys, unexpected keys, tensor shape mismatches, or checkpoint tensor channels. That requires deserializing each actual artifact into the **same** partial-hardlock instantiated model. This is particularly important because the loader intentionally uses `model.load_state_dict(..., strict=False)` (`model_loader.py:269-277`), which can otherwise allow a partially matched checkpoint to proceed.

Therefore the correct status is:

- Official baseline under this custom partial-hardlock entry: **feasible candidate, runtime compatibility unverified**.
- Stage 1 epoch 08 under this entry: **feasible candidate, runtime compatibility unverified**.
- Stage 2 epoch 04 under this entry: **feasible candidate, runtime compatibility unverified**.

No code path should be changed to coerce a failed model load. A future GPU preflight must record, per checkpoint: exact local artifact hash, `missing_keys`, `unexpected_keys`, every tensor shape mismatch, resolved `config.yaml`, and one non-output-producing construction/load result before any generation.

## Parameter diff

The intended controlled request/config has no mathematical parameter difference among checkpoints. Only the fields below may differ:

| Field | Official | Stage 1 | Stage 2 | Allowed? |
| --- | --- | --- | --- |
| `checkpoint_path` | NVIDIA artifact above | Stage 1 artifact above | Stage 2 artifact above | Yes, the sole intended model difference |
| output `name` / output directory | unique label | unique label | unique label | Yes, bookkeeping only |
| all request fields, entry script, config file, experiment, decoder route | identical | identical | identical | Required |

