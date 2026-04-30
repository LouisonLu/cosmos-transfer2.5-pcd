#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpg", ".mpeg", ".m4v"}


PCD_STEM_SUFFIXES = (
	"_vis_pcd",
)


def _iter_video_files(src_dir: Path, *, allow_all_files: bool = False) -> list[Path]:
	if not src_dir.exists():
		raise FileNotFoundError(f"Source directory not found: {src_dir}")
	if not src_dir.is_dir():
		raise NotADirectoryError(f"Source path is not a directory: {src_dir}")

	files: list[Path] = []
	for entry in src_dir.iterdir():
		if entry.name.startswith("."):
			continue
		if not entry.is_file():
			continue
		if allow_all_files or entry.suffix.lower() in VIDEO_EXTS:
			files.append(entry)

	files.sort(key=lambda p: p.name)
	return files


def _ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def _safe_symlink(target: Path, link_path: Path, *, overwrite: bool) -> None:
	if link_path.exists() or link_path.is_symlink():
		if link_path.is_symlink():
			existing = os.readlink(link_path)
			if Path(existing).resolve() == target.resolve():
				return
		if not overwrite:
			raise FileExistsError(f"Link already exists: {link_path}")
		link_path.unlink()

	os.symlink(str(target), str(link_path))


def create_numbered_symlinks(
	*,
	src_files: list[Path],
	dest_dir: Path,
	prefix: str = "video",
	overwrite: bool = True,
) -> list[Path]:
	_ensure_dir(dest_dir)

	created: list[Path] = []
	for idx, src in enumerate(src_files, start=1):
		link_path = dest_dir / f"{prefix}{idx}.{src.suffix}"
		_safe_symlink(src, link_path, overwrite=overwrite)
		created.append(link_path)
	return created


def _pcd_key(pcd_path: Path) -> str:
	stem = pcd_path.stem
	for suffix in PCD_STEM_SUFFIXES:
		if stem.endswith(suffix):
			return stem[: -len(suffix)]
	return stem


def _rgb_key(rgb_path: Path) -> str:
	return rgb_path.stem


def match_pcd_rgb_pairs(
	*,
	pcd_files: list[Path],
	rgb_files: list[Path],
) -> list[tuple[str, Path, Path]]:
	rgb_by_key: dict[str, Path] = {}
	for rgb in rgb_files:
		key = _rgb_key(rgb)
		if key not in rgb_by_key:
			rgb_by_key[key] = rgb

	pairs: list[tuple[str, Path, Path]] = []
	for pcd in pcd_files:
		key = _pcd_key(pcd)
		rgb = rgb_by_key.get(key)
		if rgb is None:
			continue
		pairs.append((key, pcd, rgb))

	pairs.sort(key=lambda t: t[0])
	return pairs


def write_dummy_captions(
	*,
	captions_dir: Path,
	video_ids: list[str],
	overwrite: bool = True,
) -> list[Path]:
	_ensure_dir(captions_dir)

	written: list[Path] = []
	for video_id in video_ids:
		out_path = captions_dir / f"{video_id}.json"
		if out_path.exists() and not overwrite:
			continue

		payload = {
            "caption": "360 panorama video."
        }
		out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
		written.append(out_path)
	return written


def main(argv: list[str] | None = None) -> int:
	repo_root = Path(__file__).resolve().parents[1]

	parser = argparse.ArgumentParser(
		description=(
			"Create dataset symlinks for 1561dataset (depth/videos) and create dummy caption JSONs "
			"for mydataset/captions."
		)
	)
	parser.add_argument(
		"--pcd-src",
		type=Path,
		default=Path("/workspace/pcd_videos"),
		help="Directory containing source PCD/depth videos (default: /workspace/pcd_videos)",
	)
	parser.add_argument(
		"--rgb-src",
		type=Path,
		default=Path("/workspace/rgb_videos"),
		help="Directory containing source RGB videos (default: /workspace/rgb_videos)",
	)
	parser.add_argument(
		"--depth-dest",
		type=Path,
		default=repo_root / "datasets/1561dataset/depth",
		help="Destination directory for numbered depth symlinks",
	)
	parser.add_argument(
		"--rgb-dest",
		type=Path,
		default=repo_root / "datasets/1561dataset/videos",
		help="Destination directory for numbered RGB symlinks",
	)
	parser.add_argument(
		"--captions-dest",
		type=Path,
		default=repo_root / "datasets/1561dataset/captions",
		help="Destination directory for dummy caption JSON files",
	)
	parser.add_argument(
		"--allow-all-files",
		action="store_true",
		help="Treat all files in src dirs as videos (ignore extension filtering)",
	)
	parser.add_argument(
		"--no-overwrite",
		action="store_true",
		help="Do not overwrite existing symlinks/files",
	)
	args = parser.parse_args(argv)

	overwrite = not args.no_overwrite

	pcd_files = _iter_video_files(args.pcd_src, allow_all_files=args.allow_all_files)
	rgb_files = _iter_video_files(args.rgb_src, allow_all_files=args.allow_all_files)

	if not pcd_files:
		print(f"No video files found in {args.pcd_src}", file=sys.stderr)
		return 2
	if not rgb_files:
		print(f"No video files found in {args.rgb_src}", file=sys.stderr)
		return 2

	pairs = match_pcd_rgb_pairs(pcd_files=pcd_files, rgb_files=rgb_files)
	if not pairs:
		print(
			"No matching PCD/RGB pairs found. Expected PCD stems like '<key>_vis_pcd' and RGB stems like '<key>'.",
			file=sys.stderr,
		)
		return 2

	pcd_matched = [pcd for _, pcd, _ in pairs]
	rgb_matched = [rgb for _, _, rgb in pairs]

	create_numbered_symlinks(src_files=pcd_matched, dest_dir=args.depth_dest, overwrite=overwrite)
	create_numbered_symlinks(src_files=rgb_matched, dest_dir=args.rgb_dest, overwrite=overwrite)

	n = len(pairs)
	video_ids = [f"video{i}" for i in range(1, n + 1)]
	write_dummy_captions(captions_dir=args.captions_dest, video_ids=video_ids, overwrite=overwrite)

	print(f"Matched {n} PCD/RGB pairs")
	print(f"Created/updated {n} depth symlinks in {args.depth_dest}")
	print(f"Created/updated {n} rgb symlinks in {args.rgb_dest}")
	print(f"Created/updated {n} dummy caption JSONs in {args.captions_dest}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

