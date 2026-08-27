# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from cosmos_transfer2._src.transfer2.utils.target_loss_mask import (
    align_valid_mask_to_augmented_video,
    build_safe_latent_valid_mask,
    masked_mse_sums_per_sample,
)


def test_align_valid_mask_uses_nearest_resize_and_zero_padding() -> None:
    source = torch.tensor([[[1, 1, 0, 0], [1, 1, 0, 0]]], dtype=torch.float32)

    aligned = align_valid_mask_to_augmented_video(source, image_size=(6, 8, 4, 8))

    assert aligned.shape == (1, 1, 6, 8)
    assert torch.count_nonzero(aligned[:, :, 0]) == 0
    assert torch.count_nonzero(aligned[:, :, -1]) == 0
    assert torch.all(aligned[:, :, 1:5, :4] == 1)
    assert torch.all(aligned[:, :, 1:5, 4:] == 0)


def test_align_valid_mask_rejects_aspect_ratio_mismatch() -> None:
    source = torch.ones(1, 4, 4)

    with pytest.raises(ValueError, match="aspect ratios do not match"):
        align_valid_mask_to_augmented_video(source, image_size=(4, 8, 4, 8))


def test_safe_latent_mask_invalidates_any_overlapping_latent_cell() -> None:
    pixel_mask = torch.ones(1, 1, 1, 8, 8)
    pixel_mask[:, :, :, 0, 0] = 0

    latent_mask = build_safe_latent_valid_mask(
        pixel_mask,
        latent_shape_T_H_W=(1, 2, 2),
        spatial_guard_px=0,
        temporal_guard_frames=0,
        min_valid_ratio=0.0,
    )

    expected = torch.tensor([[[[[0, 1], [1, 1]]]]], dtype=torch.float32)
    assert torch.equal(latent_mask, expected)


def test_safe_latent_mask_rejects_zero_valid_coverage() -> None:
    with pytest.raises(ValueError, match="too little valid latent coverage"):
        build_safe_latent_valid_mask(
            torch.zeros(1, 1, 2, 8, 8),
            latent_shape_T_H_W=(1, 2, 2),
            spatial_guard_px=0,
            temporal_guard_frames=0,
            min_valid_ratio=0.001,
        )


def test_masked_mse_ignores_invalid_errors_and_zeroes_their_gradients() -> None:
    prediction = torch.zeros(1, 2, 1, 1, 2, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[:, :, :, :, 0] = 2
    target[:, :, :, :, 1] = 100
    valid_mask = torch.tensor([[[[[1, 0]]]]], dtype=torch.float32)

    numerator, denominator = masked_mse_sums_per_sample(prediction, target, valid_mask)
    loss = (numerator / denominator).mean()
    loss.backward()

    assert numerator.item() == 8
    assert denominator.item() == 2
    assert loss.item() == 4
    assert torch.all(prediction.grad[:, :, :, :, 0] == -2)
    assert torch.all(prediction.grad[:, :, :, :, 1] == 0)


def test_all_valid_mask_matches_original_full_latent_mse() -> None:
    prediction = torch.tensor(
        [
            [[[[0.0, 1.0]]], [[[2.0, 3.0]]]],
            [[[[4.0, 5.0]]], [[[6.0, 7.0]]]],
        ]
    )
    target = torch.zeros_like(prediction)
    valid_mask = torch.ones(2, 1, 1, 1, 2)

    numerator, denominator = masked_mse_sums_per_sample(prediction, target, valid_mask)
    masked_loss = numerator / denominator
    original_loss = (prediction - target).square().flatten(1).mean(dim=1)

    assert torch.allclose(masked_loss, original_loss)
