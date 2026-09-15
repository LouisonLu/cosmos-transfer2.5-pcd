#!/usr/bin/env python3
"""Run a directory of RGB/PCD/mask/prompt samples through a Cosmos inference script.

This runner deliberately has no download or upload behavior.  It prepares one
JSONL request file and launches the chosen Cosmos inference entry point once,
so the model stays loaded for the full batch.  The expected direct-child layout
under ``data_root`` is::

    rgb_videos/<stem>.mp4
    pcd_videos/<stem>_pcd.mp4
    mask/<stem>_mask.mp4
    prompts/<stem>_prompt.json

For a previously prepared benchmark input directory, pass ``--layout flat``
instead.  Its direct-child layout is::

    <stem>.mp4
    <stem>_pcd.mp4
    <stem>_mask.mp4
    <stem>_prompt.json

``--mode no-video-path`` targets the current partial-hardlock/no-video-path
entry point.  It creates a masked RGB frame 0 image context, uses the PCD
video as depth control, and can hard-lock the white region of the mask.
``--mode standard-video`` also supports the conventional Cosmos request shape
that includes ``video_path``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--"
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class LiveProgress:
    """The compact Pi3-style terminal dashboard for a Cosmos batch."""

    SPINNER = ("|", "/", "-", "\\")

    def __init__(self, total: int, default_steps: int):
        self.total = total
        self.default_steps = default_steps
        self.completed = 0
        self.successes = 0
        self.failures = 0
        self.durations: list[float] = []
        self.current_scene = "waiting"
        self.current_step = 0
        self.current_step_total = default_steps
        self.current_started_at = time.monotonic()
        self.started_at = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._drawn = False
        self._frame = 0
        self._interactive = sys.stderr.isatty()

    def start(self) -> None:
        if self._interactive:
            self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
            self._thread.start()
        else:
            self._draw(force_line=True)

    def begin_scene(self, scene_label: str) -> None:
        with self._lock:
            self.current_scene = scene_label
            self.current_step = 0
            self.current_step_total = self.default_steps
            self.current_started_at = time.monotonic()
        if not self._interactive:
            self._draw(force_line=True)

    def update_step(self, step: int, total_steps: int) -> None:
        with self._lock:
            self.current_step = max(0, step)
            self.current_step_total = max(1, total_steps)
        if not self._interactive:
            self._draw(force_line=True)

    def finish_scene(self, succeeded: bool, duration: float) -> None:
        with self._lock:
            self.completed += 1
            self.successes += int(succeeded)
            self.failures += int(not succeeded)
            self.durations.append(duration)
            self.current_scene = "selecting next sample" if self.completed < self.total else "complete"
            self.current_step = self.current_step_total
        if not self._interactive:
            self._draw(force_line=True)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._interactive and self._drawn:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def _refresh_loop(self) -> None:
        while not self._stop.is_set():
            self._draw()
            self._stop.wait(0.2)
        self._draw()

    def _snapshot(self) -> tuple[int, int, int, Optional[float], str, int, int, float, float, int]:
        with self._lock:
            average = sum(self.durations) / len(self.durations) if self.durations else None
            return (
                self.completed,
                self.successes,
                self.failures,
                average,
                self.current_scene,
                self.current_step,
                self.current_step_total,
                time.monotonic() - self.current_started_at,
                time.monotonic() - self.started_at,
                self._frame,
            )

    def _draw(self, force_line: bool = False) -> None:
        completed, successes, failures, average, current, step, step_total, sample_elapsed, elapsed, frame = self._snapshot()
        self._frame += 1
        width = 26
        filled = round(width * completed / self.total)
        bar = "#" * filled + "-" * (width - filled)
        percent = 100 * completed / self.total
        eta = None if average is None else average * (self.total - completed)
        spinner = self.SPINNER[frame % len(self.SPINNER)]
        short_current = current if len(current) <= 64 else f"{current[:28]}...{current[-33:]}"
        line1 = (
            f"Cosmos inference {spinner} [{bar}] {completed}/{self.total} ({percent:5.1f}%) "
            f"success={successes} failed={failures}"
        )
        line2 = (
            f"running: {short_current} | sample {format_duration(sample_elapsed)} | elapsed {format_duration(elapsed)} | "
            f"avg {format_duration(average)} | ETA {format_duration(eta)}"
        )
        step_width = 26
        step_filled = round(step_width * min(step, step_total) / step_total)
        step_bar = "#" * step_filled + "-" * (step_width - step_filled)
        line3 = f"steps: [{step_bar}] {min(step, step_total)}/{step_total}"

        if self._interactive:
            if self._drawn:
                sys.stderr.write("\r\033[2K\033[1A\r\033[2K\033[1A\r\033[2K")
            sys.stderr.write(f"{line1}\n{line2}\n{line3}")
            sys.stderr.flush()
            self._drawn = True
        elif force_line:
            print(f"{line1}\n{line2}\n{line3}", flush=True)


@dataclass(frozen=True)
class Sample:
    stem: str
    name: str
    rgb: Path
    pcd: Path
    prompt: Path
    mask: Optional[Path]
    request: Path
    output: Path
    work_dir: Path


def parse_prompt(path: Path) -> str:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read prompt JSON {path}: {error}") from error

    if not isinstance(payload, dict):
        raise ValueError(f"Prompt JSON must be an object: {path}")
    for key in ("caption", "text", "prompt"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"Prompt JSON has no non-empty caption, text, or prompt: {path}")


def resolve_from_cosmos_root(cosmos_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else cosmos_root / value


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty {label}: {path}")


def run_ffmpeg(command: list[str], timeout_seconds: int, scene: str) -> None:
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"FFmpeg timed out after {timeout_seconds}s for {scene}") from error
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.splitlines()[-12:])
        raise RuntimeError(f"FFmpeg failed for {scene}, exit={completed.returncode}\n{tail}")


def extract_first_frame(ffmpeg_bin: str, video: Path, destination: Path, timeout_seconds: int, scene: str) -> None:
    run_ffmpeg(
        [ffmpeg_bin, "-y", "-v", "error", "-i", str(video), "-frames:v", "1", str(destination)],
        timeout_seconds,
        scene,
    )
    require_file(destination, f"extracted frame 0 for {scene}")


def make_masked_frame(
    ffmpeg_bin: str, rgb_frame: Path, mask_frame: Path, destination: Path, timeout_seconds: int, scene: str
) -> None:
    run_ffmpeg(
        [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-i",
            str(rgb_frame),
            "-i",
            str(mask_frame),
            "-filter_complex",
            "[0:v][1:v]blend=all_mode=multiply:all_opacity=1",
            "-frames:v",
            "1",
            str(destination),
        ],
        timeout_seconds,
        scene,
    )
    require_file(destination, f"masked frame 0 for {scene}")


def make_request(sample: Sample, args: argparse.Namespace, prompt: str, image_context: Path) -> dict[str, Any]:
    request: dict[str, Any] = {
        "name": sample.name,
        "prompt": prompt,
        "seed": args.seed,
        "guidance": args.guidance,
        "image_context_path": str(image_context.resolve()),
        "resolution": args.resolution,
        "num_conditional_frames": args.num_conditional_frames,
        "num_video_frames_per_chunk": args.frame_count,
        "num_steps": args.steps,
        "keep_input_resolution": True,
        "depth": {
            "control_path": str(sample.pcd.resolve()),
            "control_weight": args.depth_control_weight,
            "mask_path": None,
            "mask_prompt": None,
        },
    }
    if args.mode == "standard-video":
        request["video_path"] = str(sample.rgb.resolve())
    if args.guided_hardlock:
        if sample.mask is None:
            raise ValueError(f"Hardlock requires a mask: {sample.stem}")
        request["guided_generation_mask"] = str(sample.mask.resolve())
        request["guided_generation_mask_first_frame_only"] = not args.hardlock_all_frames
        request["guided_generation_mask_erode_px"] = args.mask_erode_px
        request["guided_generation_step_threshold"] = args.guided_step_threshold
    return request


def build_samples(args: argparse.Namespace) -> list[Sample]:
    if not args.data_root.is_dir():
        raise ValueError(f"Missing data root: {args.data_root}")
    requires_mask = args.mode == "no-video-path" or args.image_context == "masked" or args.guided_hardlock
    if args.layout == "directories":
        rgb_root = args.data_root / args.rgb_dir
        pcd_root = args.data_root / args.pcd_dir
        prompt_root = args.data_root / args.prompt_dir
        mask_root = args.data_root / args.mask_dir
        for path, label in ((rgb_root, "RGB directory"), (pcd_root, "PCD directory"), (prompt_root, "prompt directory")):
            if not path.is_dir():
                raise ValueError(f"Missing {label}: {path}")
        if requires_mask and not mask_root.is_dir():
            raise ValueError(f"Missing mask directory: {mask_root}")
        rgb_videos = sorted(path for path in rgb_root.iterdir() if path.is_file() and path.suffix.lower() == ".mp4")
        pcd_for = lambda stem: pcd_root / f"{stem}{args.pcd_suffix}.mp4"
        prompt_for = lambda stem: prompt_root / f"{stem}{args.prompt_suffix}.json"
        mask_for = lambda stem: mask_root / f"{stem}{args.mask_suffix}.mp4"
    else:
        control_suffixes = (f"{args.pcd_suffix}.mp4", f"{args.mask_suffix}.mp4")
        rgb_videos = sorted(
            path
            for path in args.data_root.iterdir()
            if path.is_file() and path.suffix.lower() == ".mp4" and not path.name.endswith(control_suffixes)
        )
        pcd_for = lambda stem: args.data_root / f"{stem}{args.pcd_suffix}.mp4"
        prompt_for = lambda stem: args.data_root / f"{stem}{args.prompt_suffix}.json"
        mask_for = lambda stem: args.data_root / f"{stem}{args.mask_suffix}.mp4"
    if not rgb_videos:
        raise ValueError(f"No direct RGB .mp4 files for {args.layout} layout: {args.data_root}")
    if args.limit is not None:
        rgb_videos = rgb_videos[: args.limit]

    samples: list[Sample] = []
    for rgb in rgb_videos:
        stem = rgb.stem
        pcd = pcd_for(stem)
        prompt = prompt_for(stem)
        mask = mask_for(stem) if requires_mask else None
        require_file(rgb, f"RGB video for {stem}")
        require_file(pcd, f"PCD video for {stem}")
        require_file(prompt, f"prompt for {stem}")
        if mask is not None:
            require_file(mask, f"mask video for {stem}")
        parse_prompt(prompt)
        name = f"{stem}{args.name_suffix}"
        work_dir = args.work_root / name
        samples.append(
            Sample(
                stem=stem,
                name=name,
                rgb=rgb,
                pcd=pcd,
                prompt=prompt,
                mask=mask,
                request=work_dir / f"{name}.json",
                output=args.output_root / f"{name}.mp4",
                work_dir=work_dir,
            )
        )
    return samples


def prepare_sample(sample: Sample, args: argparse.Namespace) -> None:
    sample.work_dir.mkdir(parents=True, exist_ok=True)
    rgb_frame = sample.work_dir / "rgb_frame0.png"
    extract_first_frame(args.ffmpeg_bin, sample.rgb, rgb_frame, args.ffmpeg_timeout_seconds, sample.stem)
    image_context = rgb_frame
    if args.image_context == "masked":
        if sample.mask is None:
            raise ValueError(f"Masked image context requires mask: {sample.stem}")
        mask_frame = sample.work_dir / "mask_frame0.png"
        masked_frame = sample.work_dir / "masked_rgb_frame0.png"
        extract_first_frame(args.ffmpeg_bin, sample.mask, mask_frame, args.ffmpeg_timeout_seconds, sample.stem)
        make_masked_frame(args.ffmpeg_bin, rgb_frame, mask_frame, masked_frame, args.ffmpeg_timeout_seconds, sample.stem)
        image_context = masked_frame
    request = make_request(sample, args, parse_prompt(sample.prompt), image_context)
    sample.request.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def last_log_lines(path: Path, count: int = 30) -> str:
    if not path.is_file():
        return ""
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:])
    except OSError:
        return ""


def write_summary(
    path: Path,
    args: argparse.Namespace,
    all_samples: list[Sample],
    planned: list[Sample],
    skipped: list[Sample],
    results: dict[str, dict[str, Any]],
    command: Optional[list[str]],
    return_code: Optional[int],
    error: Optional[str],
) -> None:
    payload = {
        "data_root": str(args.data_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "mode": args.mode,
        "all_samples": len(all_samples),
        "planned_samples": len(planned),
        "skipped_existing": [sample.name for sample in skipped],
        "command": command,
        "return_code": return_code,
        "error": error,
        "results": results,
        "samples": [asdict(sample) for sample in all_samples],
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def run_inference(
    command: list[str],
    cosmos_root: Path,
    log_path: Path,
    planned: list[Sample],
    timeout_seconds: int,
    cuda_visible_devices: str,
    steps: int,
) -> tuple[int, dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, Any]] = {}
    progress = LiveProgress(len(planned), default_steps=steps)
    names = {sample.name for sample in planned}
    outputs = {sample.name: sample.output for sample in planned}
    current_name: Optional[str] = None
    current_started: Optional[float] = None
    lock = threading.Lock()

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    environment["COSMOS_INFERENCE_PROGRESS"] = "1"

    process = subprocess.Popen(
        command,
        cwd=cosmos_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    def observe(line: str) -> None:
        nonlocal current_name, current_started
        if "Processing sample" in line:
            candidate = line.split("Processing sample", 1)[1].strip().strip(":")
            if candidate in names:
                with lock:
                    current_name = candidate
                    current_started = time.monotonic()
                progress.begin_scene(candidate)
        elif "COSMOS_INFERENCE_STEP" in line:
            step_text = line.split("COSMOS_INFERENCE_STEP", 1)[1].strip()
            try:
                step, total_steps = (int(value) for value in step_text.split("/", 1))
            except ValueError:
                return
            progress.update_step(step, total_steps)
        elif "Generated video saved" in line or "Saved generated video" in line:
            with lock:
                name = current_name
                started = current_started
                current_name = None
                current_started = None
            if name is not None and name not in results:
                duration = time.monotonic() - started if started is not None else 0.0
                output = outputs[name]
                succeeded = output.is_file() and output.stat().st_size > 0
                results[name] = {
                    "success": succeeded,
                    "duration_seconds": duration,
                    "error": None if succeeded else f"missing output after save log: {output}",
                }
                progress.finish_scene(succeeded, duration)

    def copy_output() -> None:
        assert process.stdout is not None
        with log_path.open("w", encoding="utf-8") as log_file:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                observe(line)

    progress.start()
    progress.begin_scene(planned[0].name)
    reader = threading.Thread(target=copy_output, daemon=True)
    reader.start()
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        return_code = process.wait()
        with lock:
            name = current_name
            started = current_started
        if name is not None and name not in results:
            duration = time.monotonic() - started if started is not None else 0.0
            results[name] = {"success": False, "duration_seconds": duration, "error": "inference timeout"}
            progress.finish_scene(False, duration)
    reader.join(timeout=10)

    for sample in planned:
        if sample.name in results:
            continue
        if return_code == 0 and sample.output.is_file() and sample.output.stat().st_size > 0:
            results[sample.name] = {"success": True, "duration_seconds": 0.0}
            progress.begin_scene(sample.name)
            progress.finish_scene(True, 0.0)
        else:
            results[sample.name] = {"success": False, "duration_seconds": 0.0, "error": "output missing or inference failed"}
            progress.begin_scene(sample.name)
            progress.finish_scene(False, 0.0)
    progress.stop()
    return return_code, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_root", type=Path, help="Data root containing rgb_videos/, pcd_videos/, mask/, prompts/")
    parser.add_argument("output_root", type=Path, help="Directory receiving generated .mp4 files and the batch log")
    parser.add_argument("--cosmos-root", type=Path, required=True, help="Cosmos checkout used as the child process working directory")
    parser.add_argument("--inference-script", type=Path, required=True, help="Inference .py path, relative to --cosmos-root or absolute")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--experiment", required=True, help="Cosmos experiment name")
    parser.add_argument("--config-file", type=Path, required=True, help="Cosmos config path, relative to --cosmos-root or absolute")
    parser.add_argument("--mode", choices=("no-video-path", "standard-video"), default="no-video-path")
    parser.add_argument(
        "--layout",
        choices=("directories", "flat"),
        default="directories",
        help="directories: rgb_videos/, pcd_videos/, mask/, prompts/; flat: all matching files directly under data_root",
    )
    parser.add_argument("--torchrun-bin", default="torchrun", help="torchrun executable")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--master-port", type=int, default=12345)
    parser.add_argument("--timeout-seconds", type=int, default=86400, help="Whole batch timeout")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffmpeg-timeout-seconds", type=int, default=300)
    parser.add_argument("--steps", type=int, default=35, help="Cosmos denoising/inference steps per sample")
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--resolution", default="720")
    parser.add_argument("--frame-count", type=int, default=93)
    parser.add_argument("--num-conditional-frames", type=int, default=1)
    parser.add_argument("--depth-control-weight", type=float, default=1.0)
    parser.add_argument("--image-context", choices=("masked", "rgb"), default="masked")
    parser.add_argument("--guided-hardlock", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--hardlock-all-frames", action="store_true")
    parser.add_argument("--mask-erode-px", type=int, default=8)
    parser.add_argument("--guided-step-threshold", type=int, default=25)
    parser.add_argument("--enable-guardrails", action="store_true")
    parser.add_argument("--inference-arg", action="append", default=[], help="Extra raw argument for the selected inference script; repeat as needed")
    parser.add_argument("--rgb-dir", default="rgb_videos")
    parser.add_argument("--pcd-dir", default="pcd_videos")
    parser.add_argument("--mask-dir", default="mask")
    parser.add_argument("--prompt-dir", default="prompts")
    parser.add_argument("--pcd-suffix", default="_pcd")
    parser.add_argument("--mask-suffix", default="_mask")
    parser.add_argument("--prompt-suffix", default="_prompt")
    parser.add_argument("--name-suffix", default="_cosmos_inference")
    parser.add_argument("--work-root", type=Path, default=None, help="Prepared frames and JSON requests; defaults to output_root/_work")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N sorted RGB files")
    parser.add_argument("--resume", action="store_true", help="Skip an item when output_root/<name>.mp4 already exists")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare frames/JSON requests but do not run Cosmos")
    parser.add_argument("--dry-run", action="store_true", help="Validate the input contract without creating prepared files or running Cosmos")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.nproc_per_node <= 0 or args.steps <= 0 or args.frame_count <= 0:
        parser.error("--nproc-per-node, --steps, and --frame-count must be positive")
    if args.timeout_seconds <= 0 or args.ffmpeg_timeout_seconds <= 0:
        parser.error("timeout values must be positive")
    if args.guided_hardlock is None:
        args.guided_hardlock = args.mode == "no-video-path"
    if args.work_root is None:
        args.work_root = args.output_root / "_work"
    return args


def main() -> None:
    args = parse_args()
    try:
        if not args.cosmos_root.is_dir():
            raise ValueError(f"Missing Cosmos root: {args.cosmos_root}")
        inference_script = resolve_from_cosmos_root(args.cosmos_root, args.inference_script)
        config_file = resolve_from_cosmos_root(args.cosmos_root, args.config_file)
        require_file(inference_script, "inference script")
        require_file(config_file, "config file")
        require_file(args.checkpoint, "checkpoint")
        samples = build_samples(args)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

    if args.dry_run:
        print(f"Cosmos inference dry run: {len(samples)} validated sample(s)")
        for sample in samples:
            print(sample.stem)
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "cosmos_inference_summary.json"
    planned: list[Sample] = []
    skipped: list[Sample] = []
    for sample in samples:
        if args.resume and sample.output.is_file() and sample.output.stat().st_size > 0:
            skipped.append(sample)
        else:
            planned.append(sample)

    try:
        for sample in planned:
            prepare_sample(sample, args)
    except (OSError, RuntimeError, ValueError) as error:
        message = f"Preparation failed before inference: {error}"
        print(f"ERROR: {message}", file=sys.stderr)
        write_summary(summary_path, args, samples, planned, skipped, {}, None, None, message)
        raise SystemExit(1)

    if args.prepare_only:
        write_summary(summary_path, args, samples, planned, skipped, {}, None, None, None)
        print(f"Prepared {len(planned)} sample(s). Summary: {summary_path}")
        return
    if not planned:
        write_summary(summary_path, args, samples, planned, skipped, {}, None, 0, None)
        print(f"All {len(skipped)} sample(s) already have output. Summary: {summary_path}")
        return

    spec_path = args.work_root / "batch_requests.jsonl"
    # Cosmos parses a .jsonl file one physical line at a time.  Per-sample
    # request files are intentionally pretty-printed for inspection, so load
    # and compact them before building the batch file.
    with spec_path.open("w", encoding="utf-8") as handle:
        for sample in planned:
            request = json.loads(sample.request.read_text(encoding="utf-8"))
            handle.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
    command = [
        args.torchrun_bin,
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--master_port",
        str(args.master_port),
        str(inference_script),
        "-i",
        str(spec_path),
        "-o",
        str(args.output_root),
        "--checkpoint-path",
        str(args.checkpoint),
        "--experiment",
        args.experiment,
        "--config-file",
        str(args.config_file),
    ]
    if not args.enable_guardrails:
        command.append("--disable-guardrails")
    command.extend(args.inference_arg)
    command_path = args.work_root / "cosmos_inference_command.txt"
    command_path.write_text(shlex.join(command) + "\n", encoding="utf-8")
    log_path = args.output_root / "cosmos_inference.log"
    print(f"Cosmos inference: {len(planned)} sample(s), log: {log_path}", flush=True)

    try:
        return_code, results = run_inference(
            command,
            args.cosmos_root,
            log_path,
            planned,
            args.timeout_seconds,
            args.cuda_visible_devices,
            args.steps,
        )
    except OSError as exception:
        error = f"Cannot start Cosmos inference: {exception}"
        write_summary(summary_path, args, samples, planned, skipped, {}, command, None, error)
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    failed = [name for name, result in results.items() if not result["success"]]
    error = None
    if return_code != 0:
        error = f"Cosmos inference process exited with {return_code}"
    if failed:
        suffix = f"Missing/failed outputs: {', '.join(failed)}"
        error = f"{error}; {suffix}" if error else suffix
    write_summary(summary_path, args, samples, planned, skipped, results, command, return_code, error)
    if error:
        print(f"ERROR: {error}\nLog tail:\n{last_log_lines(log_path)}", file=sys.stderr)
        raise SystemExit(1)
    print(f"PASS: {len(planned)} Cosmos inference sample(s) completed. Summary: {summary_path}")


if __name__ == "__main__":
    main()
