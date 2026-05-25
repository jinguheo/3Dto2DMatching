from __future__ import annotations

import numpy as np

from ..render.point_renderer import dilate, render_points
from .losses import m1_score, silhouette_iou


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
    search_flip_x: bool = True,
    search_flip_y: bool = False,
    invert_render: bool = False,
    score_mode: str = "m1",
) -> list[dict]:
    """Score all (yaw, pitch, flip) candidates; return sorted descending by score.

    search_flip_x: also try a horizontally mirrored render (covers cameras that
        produce a left-right flipped image relative to the CAD coordinate frame).
    search_flip_y: also try vertical flip (less common; off by default).
    """
    flip_x_variants = [False, True] if search_flip_x else [False]
    flip_y_variants = [False, True] if search_flip_y else [False]

    # Pre-dilate image features once to avoid repeated work in the inner loop.
    img_contour_dil = dilate(image_contour, 3)
    img_internal_dil = dilate(image_internal, 3) if image_internal is not None else None

    results: list[dict] = []
    for flip_x in flip_x_variants:
        for flip_y in flip_y_variants:
            for yaw in yaw_values:
                for pitch in pitch_values:
                    render_mask, render_edges, scale = render_points(
                        points, yaw, pitch, size,
                        pad=pad, point_radius=point_radius,
                        flip_x=flip_x, flip_y=flip_y,
                    )
                    if invert_render:
                        render_mask = ~render_mask
                    if score_mode == "iou":
                        score = silhouette_iou(render_mask, image_mask)
                    else:
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
                            "flip_x": flip_x,
                            "flip_y": flip_y,
                            "scale_px_per_unit": scale,
                            "tx_px": 0.0,
                            "ty_px": 0.0,
                            "score": score,
                        }
                    )

    results.sort(key=lambda r: -r["score"])
    return results
