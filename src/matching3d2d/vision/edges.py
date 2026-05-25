from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..render.point_renderer import dilate


def image_edges(
    path: Path,
    size: tuple[int, int],
    threshold_percentile: float = 88.0,
) -> np.ndarray:
    img = Image.open(path).convert("L").resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(ImageOps.autocontrast(img), dtype=np.float32)

    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
    gy[1:-1, :] = arr[2:, :] - arr[:-2, :]
    mag = np.hypot(gx, gy)

    threshold = np.percentile(mag, threshold_percentile)
    edges = mag >= max(threshold, 12.0)
    return dilate(edges, 1)
