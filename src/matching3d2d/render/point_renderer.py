from __future__ import annotations

import numpy as np

from ..geometry.transforms import rotation_matrix


def dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=0)
        out = (
            padded[:-2, :-2]
            | padded[:-2, 1:-1]
            | padded[:-2, 2:]
            | padded[1:-1, :-2]
            | padded[1:-1, 1:-1]
            | padded[1:-1, 2:]
            | padded[2:, :-2]
            | padded[2:, 1:-1]
            | padded[2:, 2:]
        )
    return out


def erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=0)
        out = (
            padded[:-2, :-2]
            & padded[:-2, 1:-1]
            & padded[:-2, 2:]
            & padded[1:-1, :-2]
            & padded[1:-1, 1:-1]
            & padded[1:-1, 2:]
            & padded[2:, :-2]
            & padded[2:, 1:-1]
            & padded[2:, 2:]
        )
    return out


def edge_from_mask(mask: np.ndarray) -> np.ndarray:
    eroded = (
        mask
        & np.roll(mask, 1, 0)
        & np.roll(mask, -1, 0)
        & np.roll(mask, 1, 1)
        & np.roll(mask, -1, 1)
    )
    return mask & ~eroded


def render_points(
    points: np.ndarray,
    yaw: float,
    pitch: float,
    size: tuple[int, int],
    pad: float,
    point_radius: int,
    roll: float = 0.0,
    scale_override: float | None = None,
    tx_px: float = 0.0,
    ty_px: float = 0.0,
    flip_x: bool = False,
    flip_y: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Orthographic render of point cloud.

    Returns (silhouette_mask, edge_mask, scale_px_per_unit).
    scale_px_per_unit is the auto-fit scale used, useful for seeding 2D refinement.

    flip_x / flip_y mirror the projected coordinates to account for cameras that
    produce a horizontally or vertically flipped image relative to the CAD model.
    """
    width, height = size
    rotated = points @ rotation_matrix(yaw, pitch, roll).T
    xy = rotated[:, :2].copy()

    if flip_x:
        xy[:, 0] = -xy[:, 0]
    if flip_y:
        xy[:, 1] = -xy[:, 1]

    if scale_override is None:
        span = np.maximum(np.ptp(xy, axis=0), 1e-6)
        scale = float(min((width * (1.0 - pad)) / span[0], (height * (1.0 - pad)) / span[1]))
    else:
        scale = float(scale_override)

    px = np.round((xy[:, 0] - xy[:, 0].mean()) * scale + width / 2.0 + tx_px).astype(np.int32)
    py = np.round(height / 2.0 - (xy[:, 1] - xy[:, 1].mean()) * scale + ty_px).astype(np.int32)

    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    mask = np.zeros((height, width), dtype=bool)
    if valid.any():
        mask[py[valid], px[valid]] = True

    if point_radius > 0:
        mask = dilate(mask, point_radius)
    mask = dilate(mask, 1)
    edges = dilate(edge_from_mask(mask), 1)
    return mask, edges, scale
