"""Align pre-rendered 3D masks to camera images using 2D transforms only.

Uses mask_top_yaw_90.png (for back/top cameras) and mask_bottom_flip.png
(for front/bottom cameras) as reference masks, then finds the best 2D
scale + translation + roll to maximize IoU with the extracted camera mask.

Usage:
    python scripts/align_mask2d_to_camera.py
    python scripts/align_mask2d_to_camera.py --mask-dir outputs/feature_viz --images camera_images
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from PIL import Image

from matching3d2d.vision.masks import extract_object_mask
from matching3d2d.vision.contours import external_contour
from matching3d2d.viz.overlays import make_contact_sheet


# ── 2D transform helpers ─────────────────────────────────────────────────────

def warp_mask(mask: np.ndarray, scale: float, tx: float, ty: float,
              roll_deg: float) -> np.ndarray:
    """Apply scale + roll + translation to a binary mask via inverse warping."""
    h, w = mask.shape
    cx, cy = w / 2.0, h / 2.0
    cos_r = np.cos(np.deg2rad(roll_deg))
    sin_r = np.sin(np.deg2rad(roll_deg))

    # Output pixel grid
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    # Inverse transform: where does each output pixel come from in the source?
    dx = xs - (cx + tx)
    dy = ys - (cy + ty)
    src_x = (cos_r * dx + sin_r * dy) / scale + cx
    src_y = (-sin_r * dx + cos_r * dy) / scale + cy

    xi = np.round(src_x).astype(np.int32)
    yi = np.round(src_y).astype(np.int32)
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    result = np.zeros((h, w), dtype=bool)
    result[valid] = mask[yi[valid], xi[valid]]
    return result


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union > 0 else 0.0


# ── Grid search ──────────────────────────────────────────────────────────────

def align_mask(ref_mask: np.ndarray, cam_mask: np.ndarray,
               scales: list[float], tx_vals: list[float],
               ty_vals: list[float], roll_vals: list[float]) -> dict:
    best = {"score": -1.0, "scale": 1.0, "tx": 0.0, "ty": 0.0, "roll": 0.0}
    for scale in scales:
        for roll in roll_vals:
            for tx in tx_vals:
                for ty in ty_vals:
                    warped = warp_mask(ref_mask, scale, tx, ty, roll)
                    score = iou(warped, cam_mask)
                    if score > best["score"]:
                        best = {"score": score, "scale": scale,
                                "tx": tx, "ty": ty, "roll": roll}
    return best


# ── Overlay ──────────────────────────────────────────────────────────────────

def save_overlay(cam_path: Path, ref_mask_warped: np.ndarray,
                 cam_mask: np.ndarray, out_path: Path) -> None:
    cam_img = np.array(
        Image.open(cam_path).convert("RGB").resize(
            (ref_mask_warped.shape[1], ref_mask_warped.shape[0]),
            Image.Resampling.LANCZOS
        )
    )
    overlay = cam_img.copy()
    # 3D mask region: tint green
    overlay[ref_mask_warped, 1] = np.minimum(
        overlay[ref_mask_warped, 1].astype(np.int32) + 80, 255
    ).astype(np.uint8)
    # Camera mask contour: red
    contour = external_contour(cam_mask)
    overlay[contour, 0] = 255
    overlay[contour, 1] = 0
    overlay[contour, 2] = 0
    Image.fromarray(overlay).save(out_path)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="2D mask-to-camera alignment using pre-rendered 3D masks."
    )
    parser.add_argument("--mask-dir", type=Path, default=Path("outputs/feature_viz"),
                        help="Directory containing mask_top_yaw_90.png and mask_bottom_flip.png")
    parser.add_argument("--images",   type=Path, default=Path("camera_images"))
    parser.add_argument("--out",      type=Path, default=Path("outputs/mask2d_align"))
    parser.add_argument("--width",    type=int,  default=576)
    parser.add_argument("--height",   type=int,  default=324)
    parser.add_argument("--scale-min",  type=float, default=0.75)
    parser.add_argument("--scale-max",  type=float, default=1.25)
    parser.add_argument("--scale-step", type=float, default=0.05)
    parser.add_argument("--tx-range",   type=int,   default=60,
                        help="Search ±tx_range pixels in x")
    parser.add_argument("--ty-range",   type=int,   default=60,
                        help="Search ±ty_range pixels in y")
    parser.add_argument("--tx-step",    type=int,   default=10)
    parser.add_argument("--ty-step",    type=int,   default=10)
    parser.add_argument("--roll-range", type=float, default=8.0)
    parser.add_argument("--roll-step",  type=float, default=2.0)
    parser.add_argument("--invert-ref", action="store_true",
                        help="Invert the reference 3D mask before matching")
    parser.add_argument("--invert-cam", action="store_true",
                        help="Invert the camera mask before matching")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    size = (args.width, args.height)

    # Build search grids
    scales = list(np.arange(args.scale_min, args.scale_max + 1e-9, args.scale_step))
    tx_vals = list(np.arange(-args.tx_range, args.tx_range + 1, args.tx_step, dtype=float))
    ty_vals = list(np.arange(-args.ty_range, args.ty_range + 1, args.ty_step, dtype=float))
    roll_vals = list(np.arange(-args.roll_range, args.roll_range + 1e-9, args.roll_step))

    n_candidates = len(scales) * len(tx_vals) * len(ty_vals) * len(roll_vals)
    print(f"Search grid: {len(scales)} scales × {len(tx_vals)} tx × {len(ty_vals)} ty "
          f"× {len(roll_vals)} roll = {n_candidates:,} candidates")

    # Load reference masks
    def load_mask(name: str) -> np.ndarray:
        path = args.mask_dir / name
        img = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
        arr = np.array(img) > 128
        return ~arr if args.invert_ref else arr

    ref_top = load_mask("mask_top_yaw_90.png")
    ref_bot = load_mask("mask_bottom_flip.png")
    print(f"Loaded top mask:    {ref_top.sum():,} px  ({ref_top.mean()*100:.1f}%)")
    print(f"Loaded bottom mask: {ref_bot.sum():,} px  ({ref_bot.mean()*100:.1f}%)")

    # Camera image groups
    all_images = sorted(args.images.glob("*.jpg")) + sorted(args.images.glob("*.png"))
    groups = {
        "top":    [p for p in all_images if "back"  in p.name.lower()],
        "bottom": [p for p in all_images if "front" in p.name.lower()],
    }
    ref_masks = {"top": ref_top, "bottom": ref_bot}

    best_rows: list[dict] = []
    overlay_paths: list[Path] = []

    for gname, img_list in groups.items():
        ref = ref_masks[gname]
        if not img_list:
            print(f"\n[{gname}] No images found — skipping")
            continue
        print(f"\n── {gname} ({len(img_list)} images) ──")

        for cam_path in img_list:
            print(f"  {cam_path.name}")
            img_out = args.out / cam_path.stem
            img_out.mkdir(parents=True, exist_ok=True)

            cam_mask = extract_object_mask(cam_path, size)
            if args.invert_cam:
                cam_mask = ~cam_mask
            coverage = float(cam_mask.sum()) / cam_mask.size
            print(f"    Camera mask: {coverage:.1%}")

            pose = align_mask(ref, cam_mask, scales, tx_vals, ty_vals, roll_vals)
            print(f"    Best: scale={pose['scale']:.2f}  tx={pose['tx']:.0f}  "
                  f"ty={pose['ty']:.0f}  roll={pose['roll']:.1f}°  IoU={pose['score']:.4f}")

            warped = warp_mask(ref, pose["scale"], pose["tx"], pose["ty"], pose["roll"])
            overlay_path = img_out / "overlay.png"
            save_overlay(cam_path, warped, cam_mask, overlay_path)
            overlay_paths.append(overlay_path)

            # Save warped mask
            Image.fromarray((warped.astype(np.uint8) * 255), mode="L").save(
                img_out / "mask_warped.png"
            )

            best_rows.append({"image": cam_path.name, "group": gname, **pose})

    # Summary CSV
    csv_path = args.out / "best_poses.csv"
    if best_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
            writer.writeheader()
            writer.writerows(best_rows)
        print(f"\nWrote {csv_path}")

    if overlay_paths:
        sheet = args.out / "contact_sheet.png"
        make_contact_sheet(overlay_paths, sheet)
        print(f"Wrote {sheet}")

    print("Done.")


if __name__ == "__main__":
    main()
