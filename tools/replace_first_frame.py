#!/usr/bin/env python3
"""Replace the first frame of a target video with the first frame of a source video."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import errno
import shutil
from fractions import Fraction


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def get_video_info(path: str) -> tuple[str, int, int]:
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,width,height",
        "-of",
        "json",
        path,
    ]
    probe = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
    info = json.loads(probe.stdout)
    stream = info["streams"][0]
    rate_str = stream["r_frame_rate"]
    width = int(stream["width"])
    height = int(stream["height"])
    return rate_str, width, height


def replace_first_frame(source_video: str, target_video: str, output_video: str | None) -> str:
    rate_str, width, height = get_video_info(target_video)
    fps = float(Fraction(rate_str))
    frame_duration = 1.0 / fps

    tmp_dir = os.path.join(os.path.dirname(target_video), ".tmp_first_frame")
    os.makedirs(tmp_dir, exist_ok=True)
    frame_png = os.path.join(tmp_dir, "first_frame.png")
    temp_output = os.path.join(tmp_dir, "replaced.mp4")

    run([
        "ffmpeg",
        "-y",
        "-i",
        source_video,
        "-frames:v",
        "1",
        frame_png,
    ])

    filter_complex = (
        f"[0:v]scale={width}:{height},format=yuv420p,setsar=1,"
        f"trim=duration={frame_duration:.10f},setpts=PTS-STARTPTS[v0];"
        f"[1:v]trim=start={frame_duration:.10f},setpts=PTS-STARTPTS,setsar=1[v1];"
        f"[v0][v1]concat=n=2:v=1:a=0[v]"
    )

    run([
        "ffmpeg",
        "-y",
        "-framerate",
        rate_str,
        "-loop",
        "1",
        "-i",
        frame_png,
        "-i",
        target_video,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        temp_output,
    ])

    final_output = output_video or target_video

    if output_video is not None:
        out_dir = os.path.dirname(final_output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    try:
        os.replace(temp_output, final_output)
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
        shutil.move(temp_output, final_output)
    return final_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_video", help="Video to take the first frame from")
    parser.add_argument("target_video", help="Video whose first frame will be replaced")
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to overwriting target video.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = replace_first_frame(args.source_video, args.target_video, args.output)
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
