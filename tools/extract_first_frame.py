import cv2
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
}

def extract_first_frame(video_path: str, output_image_path: str) -> None:
    video_path = Path(video_path)
    output_image_path = Path(output_image_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError("Could not read the first frame from the video.")

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_image_path), frame):
        raise RuntimeError(f"Could not save image to: {output_image_path}")

    print(f"Saved first frame to {output_image_path}")


def extract_first_frame_for_folder(input_dir: str, output_dir: str) -> None:
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)

    if not input_dir_path.is_dir():
        raise RuntimeError(f"Input path is not a directory: {input_dir_path}")

    video_paths = sorted(
        path
        for path in input_dir_path.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_paths:
        raise RuntimeError(f"No videos found in directory: {input_dir_path}")

    for video_path in video_paths:
        relative_path = video_path.relative_to(input_dir_path)
        output_image_path = output_dir_path / relative_path.with_suffix(".jpg")
        extract_first_frame(str(video_path), str(output_image_path))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python extract_first_frame.py INPUT OUTPUT\n"
            "  INPUT: path to a video file or a directory containing videos\n"
            "  OUTPUT: path to an output image (for file input) or a directory (for folder input)"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if input_path.is_dir():
        extract_first_frame_for_folder(str(input_path), str(output_path))
    else:
        extract_first_frame(str(input_path), str(output_path))