from __future__ import annotations

import numpy as np

from ..render.point_renderer import dilate
from ..vision.contours import external_contour


def contour_f1(
    render_mask: np.ndarray,
    image_contour: np.ndarray,
    image_contour_dilated: np.ndarray | None = None,
    dilation: int = 3,
) -> float:
    render_contour = external_contour(render_mask)
    if render_contour.sum() == 0 or image_contour.sum() == 0:
        return 0.0

    img_near = image_contour_dilated if image_contour_dilated is not None else dilate(image_contour, dilation)
    render_near = dilate(render_contour, dilation)

    precision = float((render_contour & img_near).sum()) / float(render_contour.sum())
    recall = float((image_contour & render_near).sum()) / float(image_contour.sum())
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def silhouette_iou(render_mask: np.ndarray, image_mask: np.ndarray) -> float:
    inter = float((render_mask & image_mask).sum())
    union = float((render_mask | image_mask).sum())
    return inter / union if union > 0 else 0.0


def internal_edge_f1(
    render_edges: np.ndarray,
    image_internal: np.ndarray,
    image_internal_dilated: np.ndarray | None = None,
    dilation: int = 3,
) -> float:
    if render_edges.sum() == 0 or image_internal.sum() == 0:
        return 0.0

    img_near = image_internal_dilated if image_internal_dilated is not None else dilate(image_internal, dilation)
    render_near = dilate(render_edges, dilation)

    precision = float((render_edges & img_near).sum()) / float(render_edges.sum())
    recall = float((image_internal & render_near).sum()) / float(image_internal.sum())
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def m1_score(
    render_mask: np.ndarray,
    render_edges: np.ndarray,
    image_mask: np.ndarray,
    image_contour: np.ndarray,
    image_internal: np.ndarray | None = None,
    image_contour_dilated: np.ndarray | None = None,
    image_internal_dilated: np.ndarray | None = None,
) -> float:
    """Composite M1 alignment score.

    score = 0.45 * contour_f1 + 0.35 * silhouette_iou + 0.15 * internal_edge_f1
    """
    cf1 = contour_f1(render_mask, image_contour, image_contour_dilated)
    siou = silhouette_iou(render_mask, image_mask)
    ief1 = (
        internal_edge_f1(render_edges, image_internal, image_internal_dilated)
        if image_internal is not None and image_internal.sum() > 0
        else 0.0
    )
    return 0.45 * cf1 + 0.35 * siou + 0.15 * ief1
