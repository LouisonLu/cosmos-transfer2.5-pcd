from pathlib import Path

from evaluation.auto import discover_videos, match_path, parse_labeled_root


def test_parse_labeled_root_preserves_equals_in_path() -> None:
    label, path = parse_labeled_root("stage2=/tmp/a=b")

    assert label == "stage2"
    assert path == Path("/tmp/a=b").resolve()


def test_discover_videos_is_bounded_and_ignores_surrogate(tmp_path: Path) -> None:
    root = tmp_path / "predictions"
    nested = root / "different_scene"
    nested.mkdir(parents=True)
    (root / "top.mp4").write_bytes(b"video")
    (nested / "scene.mp4").write_bytes(b"video")
    (nested / "scene_input_surrogate.mp4").write_bytes(b"video")
    deep = nested / "too_deep" / "ignored"
    deep.mkdir(parents=True)
    (deep / "deep.mp4").write_bytes(b"video")

    paths = discover_videos(root, max_depth=1)

    assert [path.name for path in paths] == ["top.mp4", "scene.mp4"]


def test_match_path_uses_longest_gt_prefix() -> None:
    candidates = {
        "scene": Path("scene.mp4"),
        "scene-special": Path("scene-special.mp4"),
    }

    assert match_path("scene-special_stage2.mp4"[:-4], candidates) == Path("scene-special.mp4")
