from __future__ import annotations

from pathlib import Path

import numpy as np


def load_xyz(
    path: Path,
    max_points: int | None = None,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns in {path}")

    points = data[:, :3]
    normals = data[:, 3:6] if data.shape[1] >= 6 else None

    if max_points and len(points) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        normals = normals[idx] if normals is not None else None

    points = points - points.mean(axis=0, keepdims=True)
    return points, normals
