# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the PwP geometry ablation (scripts/pwp_geometry/README.md)."""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from cosmos_transfer2._src.predict2.datasets.utils import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO
from cosmos_transfer2._src.transfer2.pwp_geometry.area_weight import build_sphere_area_weights
from cosmos_transfer2._src.transfer2.pwp_geometry.frame_size import (
    NATIVE_RESOLUTION,
    check_native_erp_input,
    resolution_for_frame_size,
)
from cosmos_transfer2._src.transfer2.pwp_geometry.masks import (
    build_safe_latent_valid_mask_for_vae,
    erode_guided_generation_mask_circular_width,
)
from cosmos_transfer2._src.transfer2.pwp_geometry.method import (
    STAMP_UNSET,
    PwPMethod,
    checkpoint_mismatches,
    run_name_resolver,
)
from cosmos_transfer2._src.transfer2.pwp_geometry.sphere_pos_emb import (
    SphereXYZPositionalEmbedding,
    sphere_xyz,
    token_latitude_longitude,
)
from cosmos_transfer2._src.transfer2.pwp_geometry.vae_circular import (
    circular_pad_width,
    convert_width_padding_to_circular,
)
from cosmos_transfer2._src.transfer2.utils.target_loss_mask import build_safe_latent_valid_mask

# Legacy training geometry: 1024x512 ERP -> 1280x640 -> reflection pad 32 px top/bottom -> 1280x704; 8x VAE, 2x patch.
PIXEL_H, PIXEL_W, PAD = 704, 1280, 32
# Native frame size: 1280x640, no padding -> latent 80x160 -> tokens 40x80.
NATIVE_H = 640


def _padding_mask(batch_size: int = 1, pad: int = PAD) -> torch.Tensor:
    padding_mask = torch.zeros(batch_size, 1, PIXEL_H, PIXEL_W)
    padding_mask[:, :, :pad] = 1
    padding_mask[:, :, PIXEL_H - pad :] = 1
    return padding_mask


# ---------------------------------------------------------------- method switches


def test_run_names_are_readable() -> None:
    assert run_name_resolver("stage1", "off", "zero", "off", "1280x640") == "stage1_baseline"
    assert run_name_resolver("stage1", "sphere_area", "zero", "off", "1280x640") == "stage1_baseline_arealoss"
    assert run_name_resolver("stage2", "off", "circular", "xyz", "1280x640") == "stage2_baseline_seam"
    assert run_name_resolver("stage2", "sphere_area", "circular", "xyz", "1280x640") == "stage2_baseline_arealoss_seam"
    assert run_name_resolver("stage1", "off", "circular", "off", "1280x640") == "stage1_baseline_vaecircular"
    assert run_name_resolver("stage1", "off", "zero", "off", "legacy_720") == "stage1_baseline_legacy720"
    with pytest.raises(ValueError, match=r"model\.config\.pwp_frame_size=1280x640\|legacy_720"):
        run_name_resolver("stage1", "off", "zero", "off", "unset")


# ---------------------------------------------------------------- frame size


def test_native_frame_size_adds_a_key_and_leaves_720_unchanged() -> None:
    video_720, image_720 = dict(VIDEO_RES_SIZE_INFO["720"]), dict(IMAGE_RES_SIZE_INFO["720"])
    assert resolution_for_frame_size("legacy_720") == "720"
    assert resolution_for_frame_size("1280x640") == NATIVE_RESOLUTION
    assert resolution_for_frame_size("1280x640") == NATIVE_RESOLUTION  # idempotent
    assert VIDEO_RES_SIZE_INFO["720"] == video_720 and IMAGE_RES_SIZE_INFO["720"] == image_720
    # detect_aspect_ratio puts 2:1 into "16,9"; every bucket of the native key is 1280x640.
    for table in (VIDEO_RES_SIZE_INFO, IMAGE_RES_SIZE_INFO):
        assert table[NATIVE_RESOLUTION] == {key: (1280, 640) for key in video_720}


def test_native_frame_size_refuses_non_erp_inputs() -> None:
    check_native_erp_input(512, 1024, source="video")
    check_native_erp_input(640, 1280, source="video")
    with pytest.raises(ValueError, match="not 2:1"):
        check_native_erp_input(720, 1280, source="video")


def test_native_augmentors_give_1280x640_without_padding() -> None:
    pytest.importorskip("torchvision")
    from cosmos_transfer2._src.imaginaire.datasets.webdataset.augmentors.image.padding import ReflectionPadding
    from cosmos_transfer2._src.imaginaire.datasets.webdataset.augmentors.image.resize import (
        ResizeLargestSideAspectPreserving,
    )

    sizes = {"size": VIDEO_RES_SIZE_INFO[resolution_for_frame_size("1280x640")]}
    data = {"video": torch.randint(0, 255, (3, 2, 512, 1024), dtype=torch.uint8), "aspect_ratio": "16,9"}
    data = ResizeLargestSideAspectPreserving(input_keys=["video"], args=sizes)(data)
    data = ReflectionPadding(input_keys=["video"], args=sizes)(data)
    assert data["video"].shape == (3, 2, 640, 1280)
    assert torch.count_nonzero(data["padding_mask"]) == 0


def test_unset_switch_names_the_missing_override() -> None:
    with pytest.raises(ValueError, match=r"model\.config\.pwp_vae_w_padding=zero\|circular"):
        PwPMethod(area_loss="off", vae_w_padding="unset", sphere_pos_emb="off")
    with pytest.raises(ValueError, match="Invalid"):
        PwPMethod(area_loss="latitude", vae_w_padding="zero", sphere_pos_emb="off")


def test_stamp_round_trip_and_unset() -> None:
    method = PwPMethod(area_loss="sphere_area", vae_w_padding="circular", sphere_pos_emb="xyz")
    assert PwPMethod.from_stamp(method.to_stamp()) == method
    assert PwPMethod.from_stamp(STAMP_UNSET) is None


def test_checkpoint_mismatches() -> None:
    seam = PwPMethod(area_loss="off", vae_w_padding="circular", sphere_pos_emb="xyz")
    baseline_settings = {"area_loss": "off", "vae_w_padding": "zero", "sphere_pos_emb": "off"}
    assert checkpoint_mismatches(None, baseline_settings) == []
    assert checkpoint_mismatches(seam, {"vae_w_padding": "circular", "sphere_pos_emb": "xyz"}) == []
    messages = checkpoint_mismatches(seam, baseline_settings)
    assert len(messages) == 2 and "baseline_seam" in messages[0]
    assert len(checkpoint_mismatches(None, {"vae_w_padding": "circular", "sphere_pos_emb": "xyz"})) == 2


# ---------------------------------------------------------------- sphere-area loss weights


def test_area_weights_are_cosine_latitude_and_zero_on_padding() -> None:
    weights = build_sphere_area_weights(_padding_mask(), PIXEL_H // 8)[0, 0, 0, :, 0]
    assert weights.shape == (88,)
    assert torch.count_nonzero(weights[:4]) == 0 and torch.count_nonzero(weights[84:]) == 0
    expected = torch.cos(math.pi * ((torch.arange(80, dtype=torch.float64) + 0.5) / 80 - 0.5)).float()
    assert torch.allclose(weights[4:84], expected)
    # Exact band area: 2π(sin(φ+Δ/2) − sin(φ−Δ/2)) ∝ cos φ, so the normalized weights are the area fractions.
    edges = math.pi * (0.5 - torch.arange(81, dtype=torch.float64) / 80)
    band_area = (torch.sin(edges[:-1]) - torch.sin(edges[1:])).float()
    assert torch.allclose(weights[4:84] / weights[4:84].sum(), band_area / band_area.sum(), atol=1e-6)


def test_area_weights_without_padding_cover_all_native_rows() -> None:
    weights = build_sphere_area_weights(torch.zeros(1, 1, NATIVE_H, PIXEL_W), NATIVE_H // 8)[0, 0, 0, :, 0]
    expected = torch.cos(math.pi * ((torch.arange(80, dtype=torch.float64) + 0.5) / 80 - 0.5)).float()
    assert torch.allclose(weights, expected)


def test_area_weights_match_the_existing_latitude_loss() -> None:
    latitude_loss = pytest.importorskip("cosmos_transfer2._src.transfer2.utils.latitude_loss")
    for padding_mask, latent_h in ((_padding_mask(2), 88), (torch.zeros(1, 1, 512, 1024), 64)):
        assert torch.equal(
            build_sphere_area_weights(padding_mask, latent_h),
            latitude_loss.build_latitude_weights(padding_mask, latent_h),
        )


# ---------------------------------------------------------------- circular VAE padding


def test_circular_pad_width() -> None:
    x = torch.arange(5.0).view(1, 5)
    assert circular_pad_width(x, 2).tolist() == [[3, 4, 0, 1, 2, 3, 4, 0, 1]]
    assert circular_pad_width(x, 0, 1).tolist() == [[0, 1, 2, 3, 4, 0]]


def _small_wan_vae():
    wan2pt1 = pytest.importorskip("cosmos_transfer2._src.predict2.tokenizers.wan2pt1")
    torch.manual_seed(0)
    return wan2pt1.WanVAE_(
        dim=8, z_dim=4, dim_mult=[1, 2], num_res_blocks=1, attn_scales=[], temperal_downsample=[False]
    ).eval()


def test_circular_vae_is_equivariant_to_horizontal_rolls() -> None:
    """A 360° frame rolled by k pixels must give a latent rolled by k/2 and a decode rolled by k."""
    vae = _small_wan_vae()
    counts = convert_width_padding_to_circular(vae)
    assert counts["causal_conv3d"] > 0 and counts["conv2d"] > 0 and counts["zero_pad2d"] > 0
    scale = [torch.zeros(4), torch.ones(4)]
    x = torch.randn(1, 3, 5, 16, 24)
    with torch.no_grad():
        z = vae.encode(x, scale)
        z_rolled = vae.encode(torch.roll(x, shifts=4, dims=-1), scale)
        assert torch.allclose(z_rolled, torch.roll(z, shifts=2, dims=-1), atol=1e-5)
        y = vae.decode(z, scale)
        y_rolled = vae.decode(torch.roll(z, shifts=2, dims=-1), scale)
        assert torch.allclose(y_rolled, torch.roll(y, shifts=4, dims=-1), atol=1e-5)


def test_zero_padding_vae_is_not_roll_equivariant_and_interior_is_unchanged() -> None:
    zero_vae = _small_wan_vae()
    circular_vae = _small_wan_vae()
    convert_width_padding_to_circular(circular_vae)
    scale = [torch.zeros(4), torch.ones(4)]
    x = torch.randn(1, 3, 5, 16, 48)
    with torch.no_grad():
        z_zero, z_circular = zero_vae.encode(x, scale), circular_vae.encode(x, scale)
        z_zero_rolled = zero_vae.encode(torch.roll(x, shifts=4, dims=-1), scale)
    assert not torch.allclose(z_zero_rolled, torch.roll(z_zero, shifts=2, dims=-1), atol=1e-5)
    # Only columns within the receptive field of the border can differ.
    assert not torch.allclose(z_zero, z_circular, atol=1e-5)
    assert torch.allclose(z_zero[..., 10:14], z_circular[..., 10:14], atol=1e-5)


def test_conversion_refuses_twice_and_unknown_width_padding() -> None:
    vae = _small_wan_vae()
    convert_width_padding_to_circular(vae)
    with pytest.raises(RuntimeError, match="already"):
        convert_width_padding_to_circular(vae)
    with pytest.raises(ValueError, match="Unconverted"):
        convert_width_padding_to_circular(nn.Sequential(nn.ReflectionPad2d((1, 1, 0, 0))))


# ---------------------------------------------------------------- mask guards


def _target_mask_with_invalid_right_strip(width_px: int = 16) -> torch.Tensor:
    mask = torch.ones(1, 1, 9, 64, 128)
    mask[..., 128 - width_px :] = 0
    return mask


def test_zero_guard_is_the_original_function() -> None:
    mask = _target_mask_with_invalid_right_strip()
    original = build_safe_latent_valid_mask(mask, (3, 8, 16), spatial_guard_px=8)
    assert torch.equal(build_safe_latent_valid_mask_for_vae(mask, (3, 8, 16), circular_width=False), original)


def test_circular_guard_wraps_and_matches_the_original_away_from_the_seam() -> None:
    mask = _target_mask_with_invalid_right_strip()
    zero = build_safe_latent_valid_mask(mask, (3, 8, 16), spatial_guard_px=8)
    circular = build_safe_latent_valid_mask_for_vae(mask, (3, 8, 16), circular_width=True, spatial_guard_px=8)
    assert zero[..., 0].all() and not circular[..., 0].any()  # the left border sees the invalid right strip
    assert torch.equal(zero[..., 1:], circular[..., 1:])

    centered = torch.ones(1, 1, 9, 64, 128)
    centered[..., 60:70] = 0
    assert torch.equal(
        build_safe_latent_valid_mask(centered, (3, 8, 16), spatial_guard_px=8),
        build_safe_latent_valid_mask_for_vae(centered, (3, 8, 16), circular_width=True, spatial_guard_px=8),
    )


def _original_erode(mask_C_T_H_W: torch.Tensor, erode_px: int) -> torch.Tensor:
    # Copy of ControlVideo2WorldInferencePartialHardlockNoVideoPath._erode_guided_generation_mask.
    binary_T_1_H_W = (mask_C_T_H_W[:1] >= 0.5).float().permute(1, 0, 2, 3)
    eroded = 1.0 - nn.functional.max_pool2d(1.0 - binary_T_1_H_W, 2 * erode_px + 1, stride=1, padding=erode_px)
    return eroded.permute(1, 0, 2, 3).repeat(mask_C_T_H_W.shape[0], 1, 1, 1)


def test_circular_erode_wraps_and_matches_the_original_away_from_the_seam() -> None:
    mask = torch.ones(3, 2, 32, 64)
    mask[..., 60:] = 0
    original = _original_erode(mask, 4)
    circular = erode_guided_generation_mask_circular_width(mask, 4)
    assert original[..., :4].eq(1).all() and circular[..., :4].eq(0).all()
    assert torch.equal(original[..., 4:], circular[..., 4:])
    assert torch.equal(erode_guided_generation_mask_circular_width(mask, 0), mask)


def test_original_erode_copy_is_current() -> None:
    pipeline = pytest.importorskip(
        "cosmos_transfer2._src.transfer2.inference.inference_pipeline_partial_hardlock_no_video_path"
    )
    mask = torch.ones(3, 2, 32, 64)
    mask[..., 20:30, 10:40] = 0
    erode = pipeline.ControlVideo2WorldInferencePartialHardlockNoVideoPath._erode_guided_generation_mask
    assert torch.equal(erode(mask, 4), _original_erode(mask, 4))


# ---------------------------------------------------------------- spherical XYZ position embedding


def test_token_latitudes_cover_the_content_and_mirror_the_padding() -> None:
    latitude, longitude = token_latitude_longitude(_padding_mask(), 44, 80)
    latitude = latitude[0]
    assert latitude.shape == (44,) and longitude.shape == (80,)
    # 40 content token rows (2..41) span the full latitude range; padding rows mirror rows 2/3 and 40/41.
    content = latitude[2:42]
    assert content[0] < math.pi / 2 and content[-1] > -math.pi / 2
    assert torch.allclose(content, -content.flip(0))
    assert torch.allclose(latitude[:2], latitude[2:4].flip(0))
    assert torch.allclose(latitude[42:], latitude[40:42].flip(0))
    assert torch.allclose(longitude[0] + 2 * math.pi - longitude[-1], torch.tensor(2 * math.pi / 80, dtype=torch.float64))


def test_native_token_latitudes_span_the_sphere_without_padding() -> None:
    latitude, longitude = token_latitude_longitude(torch.zeros(1, 1, NATIVE_H, PIXEL_W), 40, 80)
    expected = math.pi * (0.5 - (torch.arange(40, dtype=torch.float64) + 0.5) / 40)
    assert torch.allclose(latitude[0], expected)
    assert longitude.shape == (80,)


def test_xyz_puts_the_seam_ends_and_pole_longitudes_close() -> None:
    latitude, longitude = token_latitude_longitude(torch.zeros(1, 1, 704, 1280), 44, 80)
    xyz = sphere_xyz(latitude, longitude)[0]
    neighbor = (xyz[22, 40] - xyz[22, 41]).norm()
    assert torch.allclose((xyz[22, 0] - xyz[22, -1]).norm(), neighbor)  # across the seam = any neighbor
    assert (xyz[0, 0] - xyz[0, 40]).norm() < 0.2 < (xyz[22, 0] - xyz[22, 40]).norm()  # pole vs equator
    assert torch.allclose(xyz.norm(dim=-1), torch.ones(44, 80, dtype=torch.float64))


def test_xyz_embedding_starts_at_zero_and_rejects_spatial_split() -> None:
    embedding = SphereXYZPositionalEmbedding(model_channels=32)
    output = embedding(_padding_mask(2), 44, 80)
    assert output.shape == (2, 1, 44, 80, 32) and torch.count_nonzero(output) == 0
    with pytest.raises(ValueError, match="context-parallel"):
        embedding(_padding_mask(), 22, 80)
    horizontally_padded = torch.zeros(1, 1, 704, 1280)
    horizontally_padded[..., :16] = 1
    with pytest.raises(ValueError, match="full-width"):
        embedding(horizontally_padded, 44, 80)


class _KeyEcho(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 20

    def __getitem__(self, key):
        return torch.tensor(key)


def test_epoch_sampler_reshuffles_every_pass_and_tags_the_epoch() -> None:
    data = pytest.importorskip("cosmos_transfer2._src.transfer2.pwp_geometry.data")
    dataset = _KeyEcho()
    sampler = data.PwPEpochShuffleSampler(dataset, seed=0, yield_epoch=True)
    loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=1, num_workers=2, multiprocessing_context="fork"
    )
    epochs = [[tuple(batch[0].tolist()) for batch in loader] for _ in range(3)]
    for epoch, items in enumerate(epochs):
        assert sorted(index for index, _ in items) == list(range(20))
        assert {tag for _, tag in items} == {epoch}
    orders = [[index for index, _ in items] for items in epochs]
    assert orders[0] != orders[1] != orders[2]
    plain = data.PwPEpochShuffleSampler(dataset, seed=0, yield_epoch=False)
    assert list(plain) == orders[0] and list(plain) == orders[1]


def _tiny_mask_pool(root) -> None:
    from PIL import Image

    for family, bucket, count in (
        ("av2", "mask_10_15_percent", 40),
        ("native_fov", "waymo_open", 10),
        ("viewdrop", "waymo_open", 10),
        ("renderer_confidence", "waymo_open", 10),
        ("mixed", "waymo_open", 10),
    ):
        folder = root / "train" / family / bucket
        folder.mkdir(parents=True)
        for i in range(count):
            mask = np.zeros((512, 1024), dtype=np.uint8)
            mask[200:300, i * 20 : i * 20 + 100] = 255
            Image.fromarray(mask).save(folder / f"{family}_{i:03d}.png")


def test_mask_draw_depends_on_epoch_and_video_not_on_worker(tmp_path) -> None:
    data = pytest.importorskip("cosmos_transfer2._src.transfer2.pwp_geometry.data")
    from pathlib import Path

    _tiny_mask_pool(tmp_path)
    sampler_py = Path(__file__).resolve().parents[1] / "tools" / "mask_pool_sampler.py"

    def worker_dataset():
        dataset = object.__new__(data.PwPSingleViewTransferDatasetMask)
        dataset.mask_pool_root = str(tmp_path)
        dataset.mask_pool_sampler_py = str(sampler_py)
        dataset.mask_pool_sampler_seed = 0
        dataset.mask_pool_sampler_kwargs_json = None
        dataset._dynamic_mask_sampler = None
        dataset._dynamic_mask_sampler_worker_id = None
        return dataset

    def draw(dataset, epoch, index):
        dataset._pwp_mask_seed = data.mask_seed(dataset.mask_pool_sampler_seed, epoch, index)
        return dataset._sample_dynamic_mask()[1]["path"]

    first, second = worker_dataset(), worker_dataset()
    draws = {(epoch, index): draw(first, epoch, index) for epoch in range(2) for index in range(30)}
    # Another worker, asked in a different order, gives the same mask for the same (epoch, video).
    for key in reversed(list(draws)):
        assert draw(second, *key) == draws[key]
    assert len(set(draws.values())) > 30
    assert sum(draws[(0, i)] != draws[(1, i)] for i in range(30)) > 20
