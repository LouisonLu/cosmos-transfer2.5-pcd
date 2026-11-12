# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print every difference between two resolved training configs (the config.yaml written by --dryrun).

Used to confirm that two PwP runs (or a PwP run and the original experiment) differ only where intended.
Reports only; it never fails because of differences.

    python scripts/pwp_geometry/compare_configs.py A/config.yaml B/config.yaml
"""

import argparse
from pathlib import Path
from typing import Any

import yaml


def flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(node, dict):
        items = {}
        for key, value in node.items():
            items.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return items
    if isinstance(node, list):
        items = {}
        for index, value in enumerate(node):
            items.update(flatten(value, f"{prefix}[{index}]"))
        return items
    return {prefix: node}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_a", type=Path)
    parser.add_argument("config_b", type=Path)
    args = parser.parse_args()

    a = flatten(yaml.safe_load(args.config_a.read_text()))
    b = flatten(yaml.safe_load(args.config_b.read_text()))
    keys = sorted(a.keys() | b.keys())
    differences = [key for key in keys if a.get(key, "<missing>") != b.get(key, "<missing>")]
    print(f"A: {args.config_a}\nB: {args.config_b}\n{len(differences)} differences")
    for key in differences:
        print(f"  {key}\n    A: {a.get(key, '<missing>')}\n    B: {b.get(key, '<missing>')}")


if __name__ == "__main__":
    main()
