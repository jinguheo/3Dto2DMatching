"""Generate top/bottom/common binary masks from an XYZ point cloud.

Automatically detects the widest flat surface (dominant cluster normal) to
determine the top/bottom camera directions.  All masks are rendered at the
same top-down (pitch=0°) angle so top and bottom are spatially aligned.

Output masks (outputs/masks/ by default):
  mask_top.png            top-facing silhouette  (nz > 0.1),  yaw=0°
  mask_bottom.png         bottom-facing silhouette (nz < −0.1), yaw=0°  top-aligned
  mask_common.png         intersection of top & bottom, yaw=0°

  mask_top_cam.png        top camera view  →  mask_top  + yaw=90° CW
  mask_bottom_cam.png     bottom camera view → mask_bottom + vertical flip

  mask_common_top_cam.png common region as seen by top camera    (yaw=90°)
  mask_common_bot_cam.png common region as seen by bottom camera (vertical flip)

Usage:
    python scripts/generate_masks.py
    python scripts/generate_masks.py --xyz basic_shapes/s_rice.xyz --out outputs/masks
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from PIL import Image

from matching3d2d.io.xyz import load_xyz
from matching3d2d.geometry.transforms import rotation_matrix
from matching3d2d.render.point_renderer import dilate, edge_from_mask


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def kmeans_sphere(normals: np.ndarray, k: int = 8,
                  n_iter: int = 40, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Sign-invariant k-means on the unit sphere; returns (labels, centers)."""
    rng = np.random.default_rng(seed)
    centers = normals[rng.choice(len(normals), k, replace=False)].copy()
    labels  = np.zeros(len(normals), dtype=np.int32)
    for _ in range(n_iter):
        sim    = np.abs(normals @ centers.T)
        labels = np.argmax(sim, axis=1)
        for c in range(k):
            pts = normals[labels == c]
            if len(pts) == 0:
                continue
            m  = pts.mean(axis=0)
            nm = np.linalg.norm(m)
            centers[c] = m / max(nm, 1e-8)
    counts = np.bincount(labels, minlength=k)
    order  = np.argsort(-counts)
    remap  = np.empty(k, dtype=np.int32)
    remap[order] = np.arange(k)
    return remap[labels], centers[order]


def wide_face_normal(normals: np.ndarray, seed: int = 7) -> np.ndarray:
    """Return the unit normal of the widest flat surface (largest cluster)."""
    labels, centers = kmeans_sphere(normals, seed=seed)
    wide = centers[0].copy()
    c0_mean = normals[labels == 0].mean(axis=0)
    if float(np.dot(wide, c0_mean)) < 0:
        wide = -wide
    return wide


def render_silhouette(points: np.ndarray, yaw: float, pitch: float,
                      size: tuple[int, int], point_r: int = 1) -> np.ndarray:
    """Orthographic silhouette mask (bool H×W)."""
    w, h = size
    R   = rotation_matrix(yaw, pitch, 0.0)
    rot = points @ R.T
    xy  = rot[:, :2]
    span  = np.maximum(np.ptp(xy, axis=0), 1e-6)
    pad   = 0.12
    scale = float(min((w * (1 - pad)) / span[0], (h * (1 - pad)) / span[1]))
    px = np.round((xy[:, 0] - xy[:, 0].mean()) * scale + w / 2).astype(np.int32)
    py = np.round(h / 2 - (xy[:, 1] - xy[:, 1].mean()) * scale).astype(np.int32)
    valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    mask  = np.zeros((h, w), dtype=bool)
    if valid.any():
        mask[py[valid], px[valid]] = True
    if point_r > 0:
        mask = dilate(mask, point_r)
    return dilate(mask, 1)


def save_binary(arr: np.ndarray, path: Path) -> None:
    Image.fromarray((arr.astype(np.uint8) * 255), mode="L").save(path)
    print(f"  {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate top/bottom/common binary masks.")
    parser.add_argument("--xyz",         type=Path, default=Path("basic_shapes/s_rice.xyz"))
    parser.add_argument("--out",         type=Path, default=Path("outputs/masks"))
    parser.add_argument("--width",       type=int,  default=576)
    parser.add_argument("--height",      type=int,  default=324)
    parser.add_argument("--point-r",     type=int,  default=1)
    parser.add_argument("--normal-thr",  type=float,default=0.1,
                        help="Visibility threshold: |dot(normal, cam_dir)| > thr")
    parser.add_argument("--top-yaw",     type=float,default=90.0,
                        help="Yaw angle applied to align top mask with top camera (default 90°)")
    parser.add_argument("--seed",        type=int,  default=7)
    parser.add_argument("--max-points",  type=int,  default=None)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    size = (args.width, args.height)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading {args.xyz} ...")
    points, normals = load_xyz(args.xyz, args.max_points, args.seed)
    normals_norm = normalize_rows(normals)
    print(f"  {len(points):,} points")

    # ── Detect dominant surface normal ────────────────────────────────────────
    print("Detecting wide-face normal ...")
    wide_n = wide_face_normal(normals_norm, seed=args.seed)
    print(f"  Normal: [{wide_n[0]:+.2f} {wide_n[1]:+.2f} {wide_n[2]:+.2f}]")

    # Visibility masks (camera at ±wide_n)
    # top camera: pitch=0°,   cam_dir = R(0,0)^T @ e_z = [0,0,1]
    # bot camera: pitch=180°, cam_dir = R(0,180)^T @ e_z = [0,0,-1]
    R_top = rotation_matrix(0, 0,   0).astype(np.float64)
    R_bot = rotation_matrix(0, 180, 0).astype(np.float64)
    cam_top = R_top.T @ np.array([0.0, 0.0, 1.0])
    cam_bot = R_bot.T @ np.array([0.0, 0.0, 1.0])
    thr = args.normal_thr

    vm_top = (normals_norm.astype(np.float64) @ cam_top) > thr  # nz >  thr
    vm_bot = (normals_norm.astype(np.float64) @ cam_bot) > thr  # nz < -thr
    pts_top = points[vm_top]
    pts_bot = points[vm_bot]
    print(f"  Top-facing:    {pts_top.shape[0]:,} pts  ({vm_top.mean()*100:.1f}%)")
    print(f"  Bottom-facing: {pts_bot.shape[0]:,} pts  ({vm_bot.mean()*100:.1f}%)")

    # ── Render silhouettes (all at yaw=0°, pitch=0°: top-down, aligned) ──────
    print("\nRendering silhouettes (top-down, yaw=0°) ...")
    sil_top = render_silhouette(pts_top, 0.0, 0.0, size, args.point_r)
    sil_bot = render_silhouette(pts_bot, 0.0, 0.0, size, args.point_r)
    common  = sil_top & sil_bot

    pct = lambda a: f"{a.sum():,} px  ({a.sum()/a.size*100:.1f}%)"
    print(f"  mask_top:    {pct(sil_top)}")
    print(f"  mask_bottom: {pct(sil_bot)}")
    print(f"  mask_common: {pct(common)}")

    # ── Camera-aligned renders ────────────────────────────────────────────────
    top_yaw = args.top_yaw   # rotation to match top camera orientation
    print(f"\nRendering camera-aligned (top yaw={top_yaw:.0f}°, bottom vflip) ...")

    sil_top_cam = render_silhouette(pts_top, top_yaw, 0.0, size, args.point_r)
    sil_bot_cam = render_silhouette(pts_bot, top_yaw, 0.0, size, args.point_r)
    common_top_cam = sil_top_cam & sil_bot_cam
    common_bot_cam = np.flipud(common)

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"\nSaving to {args.out}/")
    save_binary(sil_top,        args.out / "mask_top.png")
    save_binary(sil_bot,        args.out / "mask_bottom.png")
    save_binary(common,         args.out / "mask_common.png")
    save_binary(sil_top_cam,    args.out / "mask_top_cam.png")
    save_binary(np.flipud(sil_bot), args.out / "mask_bottom_cam.png")
    save_binary(common_top_cam, args.out / "mask_common_top_cam.png")
    save_binary(common_bot_cam, args.out / "mask_common_bot_cam.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
