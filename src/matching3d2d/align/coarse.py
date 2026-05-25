from __future__ import annotations

import numpy as np

from ..render.point_renderer import dilate, render_points
from .losses import m1_score


def run_coarse_search(
    points: np.ndarray,
    image_mask: np.ndarray,
    image_contour: np.ndarray,
    image_internal: np.ndarray | None,
    yaw_values: list[float],
    pitch_values: list[float],
    size: tuple[int, int],
    pad: float = 0.12,
    point_radius: int = 1,
) -> list[dict]:
    """Score all (yaw, pitch) candidates; return sorted descending by score."""
    # Pre-dilate image features once to avoid repeated work in the inner loop.
    img_contour_dil = dilate(image_contour, 3)
    img_internal_dil = dilate(image_internal, 3) if image_internal is not None else None

    results: list[dict] = []
    for yaw in yaw_values:
        for pitch in pitch_values:
            render_mask, render_edges, scale = render_points(
                points, yaw, pitch, size, pad=pad, point_radius=point_radius
            )
            score = m1_score(
                render_mask,
                render_edges,
                image_mask,
                image_contour,
                image_internal,
                image_contour_dilated=img_contour_dil,
                image_internal_dilated=img_internal_dil,
            )
            results.append(
                {
                    "yaw_deg": yaw,
                    "pitch_deg": pitch,
                    "roll_deg": 0.0,
                    "scale_px_per_unit": scale,
                    "tx_px": 0.0,
                    "ty_px": 0.0,
                    "score": score,
                }
            )

    results.sort(key=lambda r: -r["score"])
    return results
