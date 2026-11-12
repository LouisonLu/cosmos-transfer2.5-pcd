# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Horizontal circular padding for the Wan2.1 VAE, applied to an already-loaded model.

Only the width (longitude) padding changes: the left border is padded with the rightmost columns and vice
versa. Height (poles) keeps zero padding and time keeps the causal padding, so the VAE weights, the temporal
feature cache and every interior output are unchanged. Three kinds of padding exist in the Wan2.1 VAE:

- ``CausalConv3d``: symmetric zero padding in H and W, causal in T (``_padding`` tuple);
- ``nn.Conv2d(padding=1)`` after the nearest upsampling in ``Resample``;
- ``nn.ZeroPad2d((0, 1, 0, 1))`` before the stride-2 convolution in ``Resample``.

The modules are converted in place by switching their class (parameters and buffers stay the same objects).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def circular_pad_width(x: Tensor, left: int, right: int | None = None) -> Tensor:
    """Pad the last dimension with wrapped-around columns."""
    right = left if right is None else right
    if left == 0 and right == 0:
        return x
    width = x.shape[-1]
    if left > width or right > width:
        raise ValueError(f"Circular padding ({left}, {right}) is wider than the input width {width}")
    pieces = [x[..., width - left :]] if left else []
    pieces.append(x)
    if right:
        pieces.append(x[..., :right])
    return torch.cat(pieces, dim=-1)


def _circular_causal_conv3d_forward(self, x: Tensor, cache_x: Tensor | None = None) -> Tensor:
    # Same as wan2pt1.CausalConv3d.forward except that the width padding wraps around.
    padding = list(self._padding)
    if cache_x is not None and self._padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]
    x = circular_pad_width(x, padding[0], padding[1])
    x = F.pad(x, [0, 0, *padding[2:]])
    return nn.Conv3d.forward(self, x)


def _circular_conv2d_forward(self, x: Tensor) -> Tensor:
    return nn.Conv2d.forward(self, circular_pad_width(x, self._pwp_width_padding))


class CircularWidthZeroPad2d(nn.Module):
    """``nn.ZeroPad2d`` whose left/right padding wraps around; top/bottom stay zero."""

    def __init__(self, left: int, right: int, top: int, bottom: int) -> None:
        super().__init__()
        self.left, self.right, self.top, self.bottom = left, right, top, bottom

    def forward(self, x: Tensor) -> Tensor:
        return F.pad(circular_pad_width(x, self.left, self.right), (0, 0, self.top, self.bottom))

    def extra_repr(self) -> str:
        return f"left={self.left}, right={self.right}, top={self.top}, bottom={self.bottom}"


_CIRCULAR_CLASSES: dict[type, type] = {}


def _circular_subclass(cls: type, forward) -> type:
    """A subclass of ``cls`` with a circular-width ``forward``; cached so isinstance checks stay stable."""
    if cls not in _CIRCULAR_CLASSES:
        _CIRCULAR_CLASSES[cls] = type(f"CircularWidth{cls.__name__}", (cls,), {"forward": forward})
    return _CIRCULAR_CLASSES[cls]


def _is_causal_conv3d(module: nn.Module) -> bool:
    return isinstance(module, nn.Conv3d) and hasattr(module, "_padding")


def convert_width_padding_to_circular(root: nn.Module) -> dict[str, int]:
    """Convert every width padding inside ``root`` to circular padding, in place.

    Returns the number of converted modules per kind. Raises if a padding kind that this function does not
    know remains, so a different VAE architecture cannot be converted only partially.
    """
    if getattr(root, "_pwp_circular_width", False):
        raise RuntimeError("The VAE width padding has already been converted to circular.")
    counts = {"causal_conv3d": 0, "conv2d": 0, "zero_pad2d": 0}

    for module in list(root.modules()):
        if _is_causal_conv3d(module):
            if module._padding[0] != module._padding[1]:
                raise ValueError(f"Asymmetric CausalConv3d width padding {module._padding}")
            if module._padding[0] > 0:
                module.__class__ = _circular_subclass(type(module), _circular_causal_conv3d_forward)
                counts["causal_conv3d"] += 1
        elif isinstance(module, nn.Conv2d):
            if module.padding_mode != "zeros" or isinstance(module.padding, str):
                raise ValueError(f"Unsupported Conv2d padding {module.padding!r} ({module.padding_mode})")
            pad_h, pad_w = module.padding
            if pad_w > 0:
                module._pwp_width_padding = pad_w
                module.padding = (pad_h, 0)
                module.__class__ = _circular_subclass(type(module), _circular_conv2d_forward)
                counts["conv2d"] += 1

    for parent in list(root.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.ZeroPad2d):
                left, right, top, bottom = child.padding
                if left or right:
                    setattr(parent, name, CircularWidthZeroPad2d(left, right, top, bottom))
                    counts["zero_pad2d"] += 1

    for name, module in root.named_modules():
        if isinstance(module, nn.Conv3d) and not _is_causal_conv3d(module) and module.padding[-1] > 0:
            raise ValueError(f"Unconverted Conv3d width padding at {name}: {module.padding}")
        if isinstance(module, (nn.ConstantPad2d, nn.ReflectionPad2d, nn.ReplicationPad2d)) and any(
            module.padding[:2]
        ):
            raise ValueError(f"Unconverted 2D width padding module at {name}: {module}")

    root._pwp_circular_width = True
    return counts


def convert_wan_tokenizer_to_circular(tokenizer) -> dict[str, int]:
    """Convert a ``Wan2pt1VAEInterface`` (as built from ``wan2pt1_tokenizer``) to circular width padding."""
    wan_vae = getattr(tokenizer, "model", None)
    vae_module = getattr(wan_vae, "model", None)
    if not isinstance(vae_module, nn.Module):
        raise TypeError(
            f"Expected a Wan2.1 VAE interface (tokenizer.model.model is the VAE module), got {type(tokenizer)}"
        )
    if getattr(wan_vae, "is_parallel", False):
        raise ValueError(
            "The Wan2.1 VAE spatial context parallelism splits the width across ranks and does not support "
            "circular width padding. Disable it (is_parallel=False / --enable-parallel-tokenizer off)."
        )
    return convert_width_padding_to_circular(vae_module)


__all__ = [
    "CircularWidthZeroPad2d",
    "circular_pad_width",
    "convert_wan_tokenizer_to_circular",
    "convert_width_padding_to_circular",
]
