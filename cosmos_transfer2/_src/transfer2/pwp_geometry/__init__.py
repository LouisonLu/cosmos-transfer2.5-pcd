# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PwP spherical-geometry ablation: sphere-area loss, horizontal circular VAE padding, XYZ position embedding.

Everything here is additive: the original models, VAE, DiT and inference pipelines are reused through
subclassing and are never modified. See scripts/pwp_geometry/README.md for the commands.
"""
