from __future__ import annotations

import numpy as np


def external_contour(mask: np.ndarray) -> np.ndarray:
    """Pixels on the outer boundary of a binary mask."""
    interior = (
        mask
        & np.roll(mask, 1, 0)
        & np.roll(mask, -1, 0)
        & np.roll(mask, 1, 1)
        & np.roll(mask, -1, 1)
    )
    return mask & ~interior


def internal_edges(
    all_edges: np.ndarray,
    mask: np.ndarray,
    contour: np.ndarray,
) -> np.ndarray:
    """Gradient edges that fall inside the object, excluding the outer contour."""
    return all_edges & mask & ~contour
