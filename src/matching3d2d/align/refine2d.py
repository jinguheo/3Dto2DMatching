from __future__ import annotations

import numpy as np

from ..geometry.transforms import rotation_matrix
from ..render.point_renderer import dilate, edge_from_mask
from .losses import m1_score, silhouette_iou

_SCALE_DELTAS = [-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12]
_TX_PX = [-40.0, -20.0, 0.0, 20.0, 40.0]
_TY_PX = [-40.0, -20.0, 0.0, 20.0, 40.0]
_ROLL_DEG = [-8.0, -4.0, 0.0, 4.0, 8.0]


def refine2d(
    points: np.ndarray,
    base_pose: dict,
    image_mask: np.ndarray,
    image_contour: np.ndarray,
    image_internal: np.ndarray | None,
    size: tuple[int, int],
    pad: float = 0.12,
    point_radius: int = 1,
    scale_deltas: list[float] | None = None,
    tx_values: list[float] | None = None,
    ty_values: list[float] | None = None,
    roll_values: list[float] | None = None,
    invert_render: bool = False,
    score_mode: str = "m1",
) -> dict:
    """Grid search over 2D scale, translation, and roll offsets.

    The 3D rotation (yaw, pitch) is fixed from the coarse pose.  Points are
    pre-projected once; the inner loop only applies cheap 2D transforms.
    """
    scale_deltas = scale_deltas if scale_deltas is not None else _SCALE_DELTAS
    tx_values = tx_values if tx_values is not None else _TX_PX
    ty_values = ty_values if ty_values is not None else _TY_PX
    roll_values = roll_values if roll_values is not None else _ROLL_DEG

    width, height = size
    yaw = base_pose["yaw_deg"]
    pitch = base_pose["pitch_deg"]
    base_scale = float(base_pose["scale_px_per_unit"])
    flip_x = bool(base_pose.get("flip_x", False))
    flip_y = bool(base_pose.get("flip_y", False))

    # Pre-rotate once (the expensive part), then apply the fixed flip from coarse.
    rotated = points @ rotation_matrix(yaw, pitch, 0.0).T
    xy_centered = rotated[:, :2] - rotated[:, :2].mean(axis=0, keepdims=True)
    if flip_x:
        xy_centered = xy_centered.copy()
        xy_centered[:, 0] = -xy_centered[:, 0]
    if flip_y:
        xy_centered = xy_centered.copy()
        xy_centered[:, 1] = -xy_centered[:, 1]

    # Pre-dilate image features to avoid redundant work in the inner loop.
    img_contour_dil = dilate(image_contour, 3)
    img_internal_dil = dilate(image_internal, 3) if image_internal is not None else None

    best = dict(base_pose)
    best_score = float(base_pose["score"])

    for roll in roll_values:
        if roll != 0.0:
            cos_r = float(np.cos(np.deg2rad(roll)))
            sin_r = float(np.sin(np.deg2rad(roll)))
            # 2-D rotation in model XY plane (z-up).
            xy_rot = np.column_stack(
                [
                    xy_centered[:, 0] * cos_r - xy_centered[:, 1] * sin_r,
                    xy_centered[:, 0] * sin_r + xy_centered[:, 1] * cos_r,
                ]
            ).astype(np.float32)
        else:
            xy_rot = xy_centered

        for sd in scale_deltas:
            scale = base_scale * (1.0 + sd)
            xy_scaled = xy_rot * scale  # shape (N, 2) in pixel units, centered

            for tx in tx_values:
                for ty in ty_values:
                    px_f = xy_scaled[:, 0] + (width / 2.0 + tx)
                    py_f = (height / 2.0 + ty) - xy_scaled[:, 1]

                    px_i = np.round(px_f).astype(np.int32)
                    py_i = np.round(py_f).astype(np.int32)
                    valid = (px_i >= 0) & (px_i < width) & (py_i >= 0) & (py_i < height)

                    render_mask = np.zeros((height, width), dtype=bool)
                    if valid.any():
                        render_mask[py_i[valid], px_i[valid]] = True
                    if point_radius > 0:
                        render_mask = dilate(render_mask, point_radius)
                    render_mask = dilate(render_mask, 1)
                    if invert_render:
                        render_mask = ~render_mask
                    render_edges = dilate(edge_from_mask(render_mask), 1)

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
                    if score > best_score:
                        best_score = score
                        best = {
                            "yaw_deg": yaw,
                            "pitch_deg": pitch,
                            "roll_deg": roll,
                            "flip_x": flip_x,
                            "flip_y": flip_y,
                            "scale_px_per_unit": scale,
                            "tx_px": tx,
                            "ty_px": ty,
                            "score": score,
                        }

    return best
