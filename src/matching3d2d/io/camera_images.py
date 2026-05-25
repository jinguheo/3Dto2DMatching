from __future__ import annotations

from pathlib import Path


def load_image_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.jpg"))


def infer_capture_group(path: Path) -> str:
    name = path.stem.lower()
    if "front" in name:
        return "top"
    if "back" in name:
        return "bottom"
    return "unknown"
