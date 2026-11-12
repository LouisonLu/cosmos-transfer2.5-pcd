# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PwP geometry method switches, the stamp stored in checkpoints, run names and run records."""

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

UNSET = "unset"
AREA_LOSS_CHOICES = ("off", "sphere_area")
VAE_W_PADDING_CHOICES = ("zero", "circular")
SPHERE_POS_EMB_CHOICES = ("off", "xyz")
# Frame size is a data setting shared by all methods (see frame_size.py), not part of the method or its stamp.
NATIVE_FRAME_SIZE = "1280x640"
LEGACY_FRAME_SIZE = "legacy_720"
FRAME_SIZE_CHOICES = (NATIVE_FRAME_SIZE, LEGACY_FRAME_SIZE)

# Overrides that select each switch on the training / inference command line.
OVERRIDE_KEYS = {
    "area_loss": "model.config.pwp_area_loss",
    "vae_w_padding": "model.config.pwp_vae_w_padding",
    "sphere_pos_emb": "model.config.pwp_sphere_pos_emb",
    "frame_size": "model.config.pwp_frame_size",
}
_CHOICES = {
    "area_loss": AREA_LOSS_CHOICES,
    "vae_w_padding": VAE_W_PADDING_CHOICES,
    "sphere_pos_emb": SPHERE_POS_EMB_CHOICES,
    "frame_size": FRAME_SIZE_CHOICES,
}

# Checkpoint stamp: [version, area_loss, vae_w_padding, sphere_pos_emb] as indices into the choices.
# A freshly initialized network holds STAMP_UNSET, so a checkpoint without a stamp leaves it unchanged.
STAMP_VERSION = 1
STAMP_LENGTH = 4
STAMP_UNSET = (-1,) * STAMP_LENGTH


def check_choice(field: str, value: str) -> str:
    """Return ``value`` if it is a valid choice for ``field``; explain the missing override otherwise."""
    choices = _CHOICES[field]
    if value == UNSET:
        raise ValueError(
            f"PwP setting '{field}' is not set. Add {OVERRIDE_KEYS[field]}={'|'.join(choices)} to the command."
        )
    if value not in choices:
        raise ValueError(f"Invalid {OVERRIDE_KEYS[field]}={value!r}; expected one of {choices}.")
    return value


@dataclass(frozen=True)
class PwPMethod:
    """The three geometry switches of one run. Stage 2's target loss mask is a property of the stage, not here."""

    area_loss: str
    vae_w_padding: str
    sphere_pos_emb: str

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            check_choice(field, value)

    @property
    def uses_area_loss(self) -> bool:
        return self.area_loss == "sphere_area"

    @property
    def uses_circular_vae(self) -> bool:
        return self.vae_w_padding == "circular"

    @property
    def uses_sphere_pos_emb(self) -> bool:
        return self.sphere_pos_emb == "xyz"

    def run_name(self) -> str:
        """Readable method name, e.g. ``baseline``, ``baseline_arealoss``, ``baseline_arealoss_seam``."""
        parts = ["baseline"]
        if self.uses_area_loss:
            parts.append("arealoss")
        if self.uses_circular_vae and self.uses_sphere_pos_emb:
            parts.append("seam")
        elif self.uses_circular_vae:
            parts.append("vaecircular")
        elif self.uses_sphere_pos_emb:
            parts.append("xyzpos")
        return "_".join(parts)

    def to_stamp(self) -> list[int]:
        return [
            STAMP_VERSION,
            AREA_LOSS_CHOICES.index(self.area_loss),
            VAE_W_PADDING_CHOICES.index(self.vae_w_padding),
            SPHERE_POS_EMB_CHOICES.index(self.sphere_pos_emb),
        ]

    @classmethod
    def from_stamp(cls, stamp: Sequence[int]) -> Optional["PwPMethod"]:
        """Decode a checkpoint stamp; ``None`` means the checkpoint was not trained by this pipeline."""
        stamp = tuple(int(value) for value in stamp)
        if stamp == STAMP_UNSET:
            return None
        if len(stamp) != STAMP_LENGTH or stamp[0] != STAMP_VERSION:
            raise ValueError(f"Unknown PwP checkpoint stamp {list(stamp)}")
        return cls(
            area_loss=AREA_LOSS_CHOICES[stamp[1]],
            vae_w_padding=VAE_W_PADDING_CHOICES[stamp[2]],
            sphere_pos_emb=SPHERE_POS_EMB_CHOICES[stamp[3]],
        )

    def as_dict(self) -> dict[str, str]:
        return {**asdict(self), "run_name": self.run_name()}


def checkpoint_mismatches(
    checkpoint_method: Optional[PwPMethod],
    expected: dict[str, str],
) -> list[str]:
    """Describe every ``expected`` switch that differs from the checkpoint's stamp.

    A checkpoint without a stamp (official weights, or runs of the original pipeline) was trained without any
    of the switches, so it only matches the ``off`` / ``zero`` settings.
    """
    if checkpoint_method is None:
        trained = PwPMethod(area_loss="off", vae_w_padding="zero", sphere_pos_emb="off")
        source = "a checkpoint without a PwP stamp (official or original-pipeline weights)"
    else:
        trained = checkpoint_method
        source = f"a checkpoint trained as {trained.run_name()}"
    messages = []
    for field, value in expected.items():
        trained_value = getattr(trained, field)
        if value != trained_value:
            messages.append(
                f"{OVERRIDE_KEYS[field]}={value}, but this is {source} ({field}={trained_value})"
            )
    return messages


def run_name_resolver(stage: str, area_loss: str, vae_w_padding: str, sphere_pos_emb: str, frame_size: str) -> str:
    """OmegaConf resolver used for ``job.name``: ``<stage>_<method run name>``, plus ``_legacy720`` for the old size."""
    method = PwPMethod(area_loss=area_loss, vae_w_padding=vae_w_padding, sphere_pos_emb=sphere_pos_emb)
    suffix = "_legacy720" if check_choice("frame_size", frame_size) == LEGACY_FRAME_SIZE else ""
    return f"{stage}_{method.run_name()}{suffix}"


def git_state(repo_dir: Path) -> dict[str, Any]:
    """Commit, dirty flag and a hash of the uncommitted changes; empty if git is unavailable."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True, timeout=30
        ).stdout

    try:
        commit = run("rev-parse", "HEAD").strip()
        diff = run("diff", "HEAD")
        untracked = run("ls-files", "--others", "--exclude-standard")
    except (OSError, subprocess.SubprocessError):
        return {}
    changes = diff + "\n--untracked--\n" + untracked
    return {
        "commit": commit,
        "dirty": bool(diff.strip() or untracked.strip()),
        "uncommitted_sha256": hashlib.sha256(changes.encode()).hexdigest() if changes.strip() else None,
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


__all__ = [
    "AREA_LOSS_CHOICES",
    "FRAME_SIZE_CHOICES",
    "LEGACY_FRAME_SIZE",
    "NATIVE_FRAME_SIZE",
    "OVERRIDE_KEYS",
    "PwPMethod",
    "SPHERE_POS_EMB_CHOICES",
    "STAMP_LENGTH",
    "STAMP_UNSET",
    "UNSET",
    "VAE_W_PADDING_CHOICES",
    "append_jsonl",
    "check_choice",
    "checkpoint_mismatches",
    "git_state",
    "run_name_resolver",
]
