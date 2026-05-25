from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from ..render.point_renderer import dilate, erode


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import label as ndi_label

        labeled, n = ndi_label(mask)
        if n == 0:
            return np.zeros_like(mask, dtype=bool)
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        return (labeled == int(np.argmax(sizes))).astype(bool)
    except ImportError:
        pass

    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []

    for r, c in zip(*np.where(mask)):
        r, c = int(r), int(c)
        if visited[r, c]:
            continue
        q: deque[tuple[int, int]] = deque([(r, c)])
        visited[r, c] = True
        component: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            component.append((cr, cc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))
        if len(component) > len(best):
            best = component

    result = np.zeros_like(mask, dtype=bool)
    for r, c in best:
        result[r, c] = True
    return result


def extract_object_mask(
    path: Path,
    size: tuple[int, int],
    border_px: int = 8,
    close_iters: int = 4,
    min_area_px: int = 500,
) -> np.ndarray:
    """Segment the main object by background subtraction.

    Uses the image border region as a background sample, thresholds the
    absolute difference, morphologically closes the result, and keeps the
    largest connected component.
    """
    img = Image.open(path).convert("L").resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)

    h, w = arr.shape
    bpx = min(border_px, h // 4, w // 4)
    border_vals = np.concatenate(
        [
            arr[:bpx, :].ravel(),
            arr[-bpx:, :].ravel(),
            arr[:, :bpx].ravel(),
            arr[:, -bpx:].ravel(),
        ]
    )
    bg_median = float(np.median(border_vals))

    diff = np.abs(arr - bg_median)
    threshold = np.percentile(diff, 55)
    raw_mask = diff > max(float(threshold), 8.0)

    closed = dilate(raw_mask, close_iters)
    closed = erode(closed, close_iters)

    largest = _largest_connected_component(closed)
    if int(largest.sum()) < min_area_px:
        return raw_mask.astype(bool)

    return largest
