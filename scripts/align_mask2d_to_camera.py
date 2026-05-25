"""Two-stage 2D alignment: pre-rendered 3D mask → camera image.

Stage 1 — Contour boundary matching (initial alignment):
  - Extract external contour from camera image (input)
  - Extract external contour from 3D reference mask
  - Estimate (scale, tx, ty) from contour centroid + bounding box

Stage 2 — Mask overlap refinement:
  - Small grid search around Stage 1 result
  - Maximize IoU(warped_ref_mask, camera_mask)

Reference masks (from outputs/feature_viz/):
  mask_top_yaw_90.png   → camera_back_*   (top view, already globally aligned)
  mask_bottom_flip.png  → camera_front_*  (bottom view, already globally aligned)

Usage:
    python scripts/align_mask2d_to_camera.py
    python scripts/align_mask2d_to_camera.py --no-refine    # Stage 1 only
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


# ── helpers ──────────────────────────────────────────────────────────────────

def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union > 0 else 0.0


def apply_scale_roll(mask: np.ndarray, scale: float, roll_deg: float) -> np.ndarray:
    """Scale + rotate a binary mask around image center (PIL AFFINE, C-speed)."""
    h, w = mask.shape
    img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    cx, cy = w / 2.0, h / 2.0
    cos_r = np.cos(np.deg2rad(roll_deg))
    sin_r = np.sin(np.deg2rad(roll_deg))
    # PIL AFFINE is the INVERSE transform: src = M @ dst
    a = cos_r / scale;   b = sin_r / scale
    d = -sin_r / scale;  e = cos_r / scale
    c = cx * (1 - a) - cy * b
    f = cy * (1 - e) - cx * d
    warped = img.transform((w, h), Image.AFFINE, (a, b, c, d, e, f),
                           resample=Image.NEAREST)
    return np.array(warped) > 128


def shift_mask(mask: np.ndarray, tx: int, ty: int) -> np.ndarray:
    """Integer shift via np.roll, then zero out wrapped edges."""
    h, w = mask.shape
    out = np.roll(np.roll(mask, ty, axis=0), tx, axis=1)
    if ty > 0:
        out[:ty, :] = False
    elif ty < 0:
        out[ty:, :] = False
    if tx > 0:
        out[:, :tx] = False
    elif tx < 0:
        out[:, tx:] = False
    return out


# ── Stage 1: Contour boundary matching ──────────────────────────────────────

def stage1_contour_align(ref_mask: np.ndarray, cam_mask: np.ndarray,
                         size: tuple[int, int]) -> dict:
    """Estimate (scale, tx, ty) from contour centroid + bounding box."""
    w, h = size
    cx, cy = w / 2.0, h / 2.0

    ref_contour = external_contour(ref_mask)
    cam_contour = external_contour(cam_mask)

    ref_pts = np.argwhere(ref_contour)  # (N, 2) → (y, x)
    cam_pts = np.argwhere(cam_contour)
    if len(ref_pts) == 0 or len(cam_pts) == 0:
        return {"scale": 1.0, "tx": 0.0, "ty": 0.0,
                "ref_n": int(len(ref_pts)), "cam_n": int(len(cam_pts))}

    ref_cy_, ref_cx_ = ref_pts.mean(axis=0)
    cam_cy_, cam_cx_ = cam_pts.mean(axis=0)

    # Robust bbox via 2-98 percentile (mitigates outlier boundary pixels).
    ref_lo = np.percentile(ref_pts, 2,  axis=0)
    ref_hi = np.percentile(ref_pts, 98, axis=0)
    cam_lo = np.percentile(cam_pts, 2,  axis=0)
    cam_hi = np.percentile(cam_pts, 98, axis=0)
    ref_size = np.maximum(ref_hi - ref_lo, 1.0)
    cam_size = np.maximum(cam_hi - cam_lo, 1.0)

    # Use geometric mean of (h, w) ratios so scale is direction-agnostic.
    scale = float(np.sqrt(cam_size[0] * cam_size[1] /
                          (ref_size[0] * ref_size[1])))

    # After scale around image center: centroid_new = scale*(centroid - c) + c
    # Then tx/ty shift to match camera centroid.
    tx = float(cam_cx_ - (scale * (ref_cx_ - cx) + cx))
    ty = float(cam_cy_ - (scale * (ref_cy_ - cy) + cy))

    return {"scale": scale, "tx": tx, "ty": ty,
            "ref_n": int(len(ref_pts)), "cam_n": int(len(cam_pts))}


# ── Stage 2: IoU refinement around Stage 1 result ───────────────────────────

def stage2_iou_refine(ref_mask: np.ndarray, cam_mask: np.ndarray,
                      init: dict,
                      scale_deltas: list[float],
                      tx_deltas: list[int],
                      ty_deltas: list[int],
                      roll_vals: list[float]) -> dict:
    init_scale = init["scale"]
    init_tx    = init["tx"]
    init_ty    = init["ty"]

    best = {"score": -1.0,
            "scale": init_scale, "roll_deg": 0.0,
            "tx_px": int(round(init_tx)), "ty_px": int(round(init_ty))}

    for ds in scale_deltas:
        scale = init_scale * (1.0 + ds)
        for roll in roll_vals:
            warped = apply_scale_roll(ref_mask, scale, roll)
            for dtx in tx_deltas:
                for dty in ty_deltas:
                    tx = int(round(init_tx + dtx))
                    ty = int(round(init_ty + dty))
                    shifted = shift_mask(warped, tx, ty)
                    score = iou(shifted, cam_mask)
                    if score > best["score"]:
                        best = {"score": score, "scale": scale,
                                "roll_deg": roll, "tx_px": tx, "ty_px": ty}
    return best


# ── Overlay ──────────────────────────────────────────────────────────────────

def save_overlay(cam_path: Path, ref_warped: np.ndarray, cam_mask: np.ndarray,
                 out_path: Path, size: tuple[int, int]) -> None:
    cam_img = np.array(
        Image.open(cam_path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    )
    ov = cam_img.copy()
    # Green tint over the warped 3D mask region
    ov[ref_warped, 1] = np.clip(
        ov[ref_warped, 1].astype(np.int32) + 80, 0, 255
    ).astype(np.uint8)
    # Red outline of camera mask
    ct = external_contour(cam_mask)
    ov[ct] = [255, 0, 0]
    # Yellow outline of warped reference mask
    ref_ct = external_contour(ref_warped)
    ov[ref_ct] = [255, 255, 0]
    Image.fromarray(ov).save(out_path)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", type=Path, default=Path("outputs/feature_viz"))
    parser.add_argument("--images",   type=Path, default=Path("camera_images"))
    parser.add_argument("--out",      type=Path, default=Path("outputs/mask2d_align"))
    parser.add_argument("--width",    type=int,  default=576)
    parser.add_argument("--height",   type=int,  default=324)
    parser.add_argument("--no-refine", action="store_true",
                        help="Skip Stage 2 (contour-only alignment)")
    parser.add_argument("--scale-deltas", type=str, default="-0.10,-0.05,0,0.05,0.10",
                        help="Scale multipliers around stage1 (comma list)")
    parser.add_argument("--tx-deltas",    type=str, default="-15,-10,-5,0,5,10,15")
    parser.add_argument("--ty-deltas",    type=str, default="-15,-10,-5,0,5,10,15")
    parser.add_argument("--roll-vals",    type=str, default="-4,-2,0,2,4")
    parser.add_argument("--invert-ref",   action="store_true",
                        help="Invert 3D reference mask polarity")
    parser.add_argument("--invert-cam",   action="store_true",
                        help="Invert camera mask polarity")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    size = (args.width, args.height)

    scale_deltas = [float(x) for x in args.scale_deltas.split(",")]
    tx_deltas    = [int(float(x)) for x in args.tx_deltas.split(",")]
    ty_deltas    = [int(float(x)) for x in args.ty_deltas.split(",")]
    roll_vals    = [float(x) for x in args.roll_vals.split(",")]
    print(f"Stage 2 refine grid: {len(scale_deltas)} scale × "
          f"{len(tx_deltas)} tx × {len(ty_deltas)} ty × {len(roll_vals)} roll "
          f"= {len(scale_deltas)*len(tx_deltas)*len(ty_deltas)*len(roll_vals)} candidates")

    def load_ref(name: str) -> np.ndarray:
        img = Image.open(args.mask_dir / name).convert("L").resize(
            size, Image.Resampling.NEAREST)
        arr = np.array(img) > 128
        return ~arr if args.invert_ref else arr

    ref = {"top":    load_ref("mask_top_cam.png"),     # rendered at yaw=-90
           "bottom": load_ref("mask_bottom_cam.png")}  # rendered at yaw=0, vflipped
    for k, m in ref.items():
        print(f"  ref {k}: {m.sum():,} px  ({m.mean()*100:.1f}%)")

    all_images = sorted(args.images.glob("*.jpg")) + sorted(args.images.glob("*.png"))
    groups = {
        "top":    [p for p in all_images if "back"  in p.name.lower()],
        "bottom": [p for p in all_images if "front" in p.name.lower()],
    }

    rows: list[dict] = []
    overlays: list[Path] = []

    for gname, img_list in groups.items():
        if not img_list:
            continue
        print(f"\n── {gname} ({len(img_list)} images) ──")
        for cam_path in img_list:
            cam_mask = extract_object_mask(cam_path, size)
            if args.invert_cam:
                cam_mask = ~cam_mask

            # Stage 1
            s1 = stage1_contour_align(ref[gname], cam_mask, size)

            # Initial IoU at Stage 1 result (no roll)
            warped0 = apply_scale_roll(ref[gname], s1["scale"], 0.0)
            shifted0 = shift_mask(warped0, int(round(s1["tx"])), int(round(s1["ty"])))
            iou0 = iou(shifted0, cam_mask)

            print(f"  {cam_path.name}")
            print(f"    Stage 1 (contour): scale={s1['scale']:.3f}  "
                  f"tx={s1['tx']:+.1f}  ty={s1['ty']:+.1f}  "
                  f"IoU={iou0:.4f}  (ref_n={s1['ref_n']} cam_n={s1['cam_n']})")

            if args.no_refine:
                final = {"scale": s1["scale"], "roll_deg": 0.0,
                         "tx_px": int(round(s1["tx"])),
                         "ty_px": int(round(s1["ty"])),
                         "score": iou0}
            else:
                final = stage2_iou_refine(
                    ref[gname], cam_mask, s1,
                    scale_deltas, tx_deltas, ty_deltas, roll_vals
                )
                print(f"    Stage 2 (IoU):     scale={final['scale']:.3f}  "
                      f"roll={final['roll_deg']:+.1f}°  "
                      f"tx={final['tx_px']:+d}  ty={final['ty_px']:+d}  "
                      f"IoU={final['score']:.4f}")

            warped = apply_scale_roll(ref[gname], final["scale"], final["roll_deg"])
            warped = shift_mask(warped, final["tx_px"], final["ty_px"])

            out_dir = args.out / cam_path.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray((warped.astype(np.uint8) * 255), "L").save(
                out_dir / "mask_warped.png")
            ov_path = out_dir / "overlay.png"
            save_overlay(cam_path, warped, cam_mask, ov_path, size)
            overlays.append(ov_path)

            rows.append({
                "image": cam_path.name, "group": gname,
                "stage1_scale": s1["scale"], "stage1_tx": s1["tx"],
                "stage1_ty": s1["ty"], "stage1_iou": iou0,
                **final,
            })

    if rows:
        csv_path = args.out / "best_poses.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
        print(f"\nWrote {csv_path}")

    if overlays:
        sheet = args.out / "contact_sheet.png"
        make_contact_sheet(overlays, sheet)
        print(f"Wrote {sheet}")

    print("Done.")


if __name__ == "__main__":
    main()
