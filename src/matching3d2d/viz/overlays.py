from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def overlay_m1(
    camera_path: Path,
    render_mask: np.ndarray,
    render_edges: np.ndarray,
    image_mask: np.ndarray,
    image_contour: np.ndarray,
    out_path: Path,
) -> None:
    """Render overlay: dimmed background, blue image contour, red CAD edges, yellow-green overlap."""
    size = (render_mask.shape[1], render_mask.shape[0])
    base = Image.open(camera_path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(base).copy()

    # Dim background to focus attention on the object.
    arr[~image_mask] = (arr[~image_mask] * 0.35).astype(np.uint8)

    # Image contour: blue.
    arr[image_contour] = (40, 140, 255)

    # Rendered CAD edges: red.
    arr[render_edges] = (255, 60, 50)

    # Overlap between image contour and CAD edges: yellow-green.
    overlap = image_contour & render_edges
    arr[overlap] = (100, 255, 80)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)


def overlay_legacy(
    camera_path: Path,
    render_edges: np.ndarray,
    image_edges_mask: np.ndarray,
    out_path: Path,
) -> None:
    """M0-style overlay: blue=image edges, red=CAD edges, yellow=overlap."""
    size = (render_edges.shape[1], render_edges.shape[0])
    base = Image.open(camera_path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(base).copy()
    arr[image_edges_mask] = (40, 180, 255)
    arr[render_edges] = (255, 70, 50)
    arr[image_edges_mask & render_edges] = (255, 230, 60)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)


def save_debug_mask(
    camera_path: Path,
    image_mask: np.ndarray,
    image_contour: np.ndarray,
    out_path: Path,
) -> None:
    """Green tint over detected object region, blue outline on contour."""
    size = (image_mask.shape[1], image_mask.shape[0])
    base = Image.open(camera_path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(base, dtype=np.float32).copy()

    tint = np.array([0.0, 160.0, 0.0], dtype=np.float32)
    arr[image_mask] = arr[image_mask] * 0.55 + tint * 0.45
    arr = arr.clip(0, 255).astype(np.uint8)
    arr[image_contour] = (40, 140, 255)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)


def save_debug_edges(
    camera_path: Path,
    image_mask: np.ndarray,
    image_contour: np.ndarray,
    image_internal: np.ndarray,
    out_path: Path,
) -> None:
    """Blue outer contour, orange internal edges; background dimmed."""
    size = (image_mask.shape[1], image_mask.shape[0])
    base = Image.open(camera_path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(base).copy()

    arr[~image_mask] = (arr[~image_mask] * 0.35).astype(np.uint8)
    arr[image_internal] = (255, 140, 40)   # orange
    arr[image_contour] = (40, 140, 255)    # blue (draw on top of internal)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)


def save_debug_coarse(
    camera_path: Path,
    coarse_renders: list[tuple[np.ndarray, np.ndarray, dict]],
    out_path: Path,
    thumb_width: int = 320,
) -> None:
    """Contact sheet of top-K coarse candidate overlays.

    coarse_renders: list of (render_mask, render_edges, pose_dict) tuples.
    """
    size = (coarse_renders[0][0].shape[1], coarse_renders[0][0].shape[0])
    base_img = Image.open(camera_path).convert("RGB").resize(size, Image.Resampling.LANCZOS)

    thumbs = []
    for rank, (render_mask, render_edges, pose) in enumerate(coarse_renders, start=1):
        arr = np.asarray(base_img).copy()
        arr[~render_mask] = (arr[~render_mask] * 0.5).astype(np.uint8)
        arr[render_edges] = (255, 60, 50)

        ratio = thumb_width / size[0]
        thumb = Image.fromarray(arr).resize(
            (thumb_width, int(size[1] * ratio)), Image.Resampling.LANCZOS
        )

        label_h = 36
        canvas = Image.new("RGB", (thumb.width, thumb.height + label_h), (25, 25, 25))
        canvas.paste(thumb, (0, 0))
        draw = ImageDraw.Draw(canvas)
        flip_str = ""
        if pose.get("flip_x"):
            flip_str += " fx"
        if pose.get("flip_y"):
            flip_str += " fy"
        label = (
            f"#{rank} yaw={pose['yaw_deg']:.0f} "
            f"pit={pose['pitch_deg']:.0f}{flip_str}  "
            f"score={pose['score']:.3f}"
        )
        draw.text((5, thumb.height + 8), label, fill=(220, 220, 100))
        thumbs.append(canvas)

    total_w = sum(t.width for t in thumbs)
    total_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (total_w, total_h), (15, 15, 15))
    x = 0
    for t in thumbs:
        sheet.paste(t, (x, 0))
        x += t.width

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def make_contact_sheet(items: list[Path], out_path: Path, thumb_width: int = 384) -> None:
    thumbs = []
    for path in items:
        img = Image.open(path).convert("RGB")
        ratio = thumb_width / img.width
        thumb = img.resize((thumb_width, int(img.height * ratio)), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb.width, thumb.height + 24), "white")
        canvas.paste(thumb, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, thumb.height + 5), path.parent.name[:50], fill=(20, 20, 20))
        thumbs.append(canvas)

    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    w = cols * thumb_width
    h = rows * max(t.height for t in thumbs)
    sheet = Image.new("RGB", (w, h), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * thumb_width, (i // cols) * thumb.height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
