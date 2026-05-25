"""Align 3D top/bottom surface masks to camera images.

Camera assignment:
  camera_back_*  = top view  → top-facing XYZ points (nz > 0.1),  pitch=0°  fixed
  camera_front_* = bottom view → bottom-facing XYZ points (nz < -0.1), pitch=180° fixed

Coarse search: yaw 0-355° (5° steps) × flip variants  [72 candidates per image]
Refinement:    scale × tx × ty × roll  (same refine2d grid as M1)

Usage:
    python scripts/align_top_bottom_to_camera.py
    python scripts/align_top_bottom_to_camera.py --top-k 1 --no-refine
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from matching3d2d.io.xyz import load_xyz
from matching3d2d.render.point_renderer import render_points
from matching3d2d.vision.edges import image_edges
from matching3d2d.vision.masks import extract_object_mask
from matching3d2d.vision.contours import external_contour, internal_edges
from matching3d2d.align.coarse import run_coarse_search
from matching3d2d.align.refine2d import refine2d
from matching3d2d.geometry.transforms import rotation_matrix
from matching3d2d.viz.overlays import (
    overlay_m1,
    make_contact_sheet,
    save_debug_mask,
    save_debug_edges,
    save_debug_coarse,
)
from matching3d2d.viz.reports import save_pose_json


def normalize_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def dominant_normal(normals_norm: np.ndarray, k: int = 8, seed: int = 7) -> np.ndarray:
    """Return the unit normal of the widest flat surface via sign-invariant k-means."""
    rng = np.random.default_rng(seed)
    centers = normals_norm[rng.choice(len(normals_norm), k, replace=False)].copy()
    labels = np.zeros(len(normals_norm), dtype=np.int32)

    for _ in range(40):
        sim = np.abs(normals_norm @ centers.T)
        labels = np.argmax(sim, axis=1)
        for c in range(k):
            pts = normals_norm[labels == c]
            if len(pts) == 0:
                continue
            m = pts.mean(axis=0)
            nm = np.linalg.norm(m)
            centers[c] = m / max(nm, 1e-8)

    # Sort clusters by size (largest first)
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    remap = np.empty(k, dtype=np.int32)
    remap[order] = np.arange(k)
    labels = remap[labels]
    centers = centers[order]

    wide = centers[0].copy()
    # Align sign with the actual mean of cluster-0 normals
    c0_mean = normals_norm[labels == 0].mean(axis=0)
    if float(np.dot(wide, c0_mean)) < 0:
        wide = -wide
    return wide


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align 3D top/bottom masks to camera images."
    )
    parser.add_argument("--xyz",    type=Path, default=Path("basic_shapes/s_rice.xyz"))
    parser.add_argument("--images", type=Path, default=Path("camera_images"))
    parser.add_argument("--out",    type=Path, default=Path("outputs/mask_align"))
    parser.add_argument("--width",  type=int,  default=576)
    parser.add_argument("--height", type=int,  default=324)
    parser.add_argument("--yaw-step", type=float, default=5.0,
                        help="Coarse yaw search step in degrees (default 5)")
    parser.add_argument("--top-k",  type=int,  default=3)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--no-flip",   action="store_true")
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--seed",   type=int,  default=7)
    # Manual yaw overrides — skip coarse search entirely for that group
    parser.add_argument("--top-yaw",    type=float, default=None,
                        help="Fixed yaw for top cameras (back). Skips coarse search.")
    parser.add_argument("--top-flip-x", action="store_true",
                        help="Force flip_x=True for top cameras")
    parser.add_argument("--bot-yaw",    type=float, default=None,
                        help="Fixed yaw for bottom cameras (front). Skips coarse search.")
    parser.add_argument("--bot-flip-x", action="store_true",
                        help="Force flip_x=True for bottom cameras")
    parser.add_argument("--invert-mask", action="store_true",
                        help="Invert camera object mask before matching (use when object is dark on light bg)")
    parser.add_argument("--invert-render", action="store_true",
                        help="Invert 3D render mask before matching (use when 3D silhouette polarity is opposite)")
    parser.add_argument("--score-mode", choices=["m1", "iou"], default="iou",
                        help="Scoring function: 'iou' = pure mask overlap (default), 'm1' = contour+iou+edge composite")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    size = (args.width, args.height)

    # ── Load XYZ ──────────────────────────────────────────────────────────────
    print(f"Loading {args.xyz} ...")
    points, normals = load_xyz(args.xyz, args.max_points, args.seed)
    normals_norm = normalize_rows(normals)
    print(f"  {len(points):,} points loaded")

    # ── Detect wide-face normal → view masks ──────────────────────────────────
    print("Detecting dominant surface normal ...")
    wide_n = dominant_normal(normals_norm, seed=args.seed)
    print(f"  Wide-face normal: [{wide_n[0]:+.2f} {wide_n[1]:+.2f} {wide_n[2]:+.2f}]")

    # Camera direction vectors for top/bottom
    # top: camera at +wide_n looking toward -wide_n  → pitch=0  (after alignment)
    # bottom: camera at -wide_n looking toward +wide_n → pitch=180
    R_top = rotation_matrix(0, 0, 0).astype(np.float64)
    R_bot = rotation_matrix(0, 180, 0).astype(np.float64)
    cam_top = R_top.T @ np.array([0.0, 0.0, 1.0])
    cam_bot = R_bot.T @ np.array([0.0, 0.0, 1.0])

    mask_top = (normals_norm.astype(np.float64) @ cam_top) > 0.1   # nz > 0.1
    mask_bot = (normals_norm.astype(np.float64) @ cam_bot) > 0.1   # nz < -0.1
    pts_top = points[mask_top]
    pts_bot = points[mask_bot]
    print(f"  Top-facing:    {pts_top.shape[0]:,} pts")
    print(f"  Bottom-facing: {pts_bot.shape[0]:,} pts")

    # ── Camera image groups ───────────────────────────────────────────────────
    all_images = sorted(args.images.glob("*.jpg")) + sorted(args.images.glob("*.png"))
    groups = {
        "top":    {"images":     [p for p in all_images if "back"  in p.name.lower()],
                   "pts":        pts_top,
                   "pitch":      0.0,
                   "label":      "back→top",
                   "manual_yaw": args.top_yaw,
                   "manual_fx":  args.top_flip_x},
        "bottom": {"images":     [p for p in all_images if "front" in p.name.lower()],
                   "pts":        pts_bot,
                   "pitch":      180.0,
                   "label":      "front→bottom",
                   "manual_yaw": args.bot_yaw,
                   "manual_fx":  args.bot_flip_x},
    }

    yaw_values = list(np.arange(0.0, 360.0, args.yaw_step))
    n_coarse = len(yaw_values) * (2 if not args.no_flip else 1)
    print(f"\nCoarse candidates per image: {n_coarse}  "
          f"(yaw 0-360°/{args.yaw_step}° steps, flip={'off' if args.no_flip else 'on'})"
          f"  [override: --top-yaw / --bot-yaw to skip]")

    best_rows: list[dict] = []
    overlay_paths: list[Path] = []

    for gname, ginfo in groups.items():
        img_list   = ginfo["images"]
        pts        = ginfo["pts"]
        pitch      = ginfo["pitch"]
        glabel     = ginfo["label"]
        manual_yaw = ginfo["manual_yaw"]
        manual_fx  = ginfo["manual_fx"]

        if not img_list:
            print(f"\n  [{glabel}] No images found — skipping")
            continue

        mode = f"fixed yaw={manual_yaw:.0f}°" if manual_yaw is not None else "coarse search"
        print(f"\n── {glabel}  (pitch={pitch:.0f}°, {pts.shape[0]:,} pts, {mode}) ──")

        for image_path in img_list:
            print(f"\n  {image_path.name}")
            img_out_dir = args.out / image_path.stem
            img_out_dir.mkdir(parents=True, exist_ok=True)

            # 2D image features
            img_all_edges = image_edges(image_path, size)
            img_mask      = extract_object_mask(image_path, size)
            if args.invert_mask:
                img_mask = ~img_mask
            img_contour   = external_contour(img_mask)
            img_internal  = internal_edges(img_all_edges, img_mask, img_contour)

            coverage = float(img_mask.sum()) / img_mask.size
            print(f"    Camera mask coverage: {coverage:.1%}")

            save_debug_mask(image_path, img_mask, img_contour,
                            img_out_dir / "debug_mask.png")
            save_debug_edges(image_path, img_mask, img_contour, img_internal,
                             img_out_dir / "debug_edges.png")

            # Coarse search — or use manual yaw if provided
            if manual_yaw is not None:
                # Skip coarse: build a single fixed pose and score it
                rm, re, scale = render_points(
                    pts, manual_yaw, pitch, size,
                    pad=0.12, point_radius=1,
                    flip_x=manual_fx, flip_y=False,
                )
                from matching3d2d.align.losses import m1_score, silhouette_iou
                from matching3d2d.render.point_renderer import dilate as _dil
                if args.invert_render:
                    rm = ~rm
                if args.score_mode == "iou":
                    score0 = silhouette_iou(rm, img_mask)
                else:
                    score0 = m1_score(rm, re, img_mask, img_contour, img_internal,
                                      image_contour_dilated=_dil(img_contour, 3),
                                      image_internal_dilated=_dil(img_internal, 3) if img_internal is not None else None)
                top_k_poses = [{
                    "yaw_deg": manual_yaw, "pitch_deg": pitch,
                    "roll_deg": 0.0, "flip_x": manual_fx, "flip_y": False,
                    "scale_px_per_unit": scale, "tx_px": 0.0, "ty_px": 0.0,
                    "score": score0,
                }]
                print(f"    Fixed pose: yaw={manual_yaw:.0f}°  score={score0:.4f}  flip_x={manual_fx}")
            else:
                coarse = run_coarse_search(
                    pts, img_mask, img_contour, img_internal,
                    yaw_values=yaw_values,
                    pitch_values=[pitch],
                    size=size,
                    point_radius=1,
                    search_flip_x=not args.no_flip,
                    search_flip_y=False,
                    invert_render=args.invert_render,
                    score_mode=args.score_mode,
                )
                best_c = coarse[0]
                print(f"    Coarse best: yaw={best_c['yaw_deg']:.0f}°  "
                      f"score={best_c['score']:.4f}  flip_x={best_c['flip_x']}")

                csv_path = img_out_dir / "coarse_scores.csv"
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(coarse[0].keys()))
                    writer.writeheader()
                    writer.writerows(coarse)

                top_k_poses = coarse[: args.top_k]
                coarse_renders = []
                for pose in top_k_poses:
                    rm, re, _ = render_points(
                        pts, pose["yaw_deg"], pose["pitch_deg"], size,
                        pad=0.12, point_radius=1,
                        flip_x=pose.get("flip_x", False),
                        flip_y=pose.get("flip_y", False),
                    )
                    coarse_renders.append((rm, re, pose))
                save_debug_coarse(image_path, coarse_renders,
                                  img_out_dir / "debug_coarse.png")

            # Refine
            if args.no_refine:
                best_pose = best_c
            else:
                refined = [
                    refine2d(pts, p, img_mask, img_contour, img_internal,
                             size, point_radius=1,
                             invert_render=args.invert_render,
                             score_mode=args.score_mode)
                    for p in top_k_poses
                ]
                best_pose = max(refined, key=lambda r: r["score"])
                print(f"    Refined:    yaw={best_pose['yaw_deg']:.0f}°  "
                      f"roll={best_pose['roll_deg']:.1f}°  "
                      f"tx={best_pose['tx_px']:.0f}  ty={best_pose['ty_px']:.0f}  "
                      f"score={best_pose['score']:.4f}  "
                      f"flip_x={best_pose['flip_x']}")

            save_pose_json(best_pose, img_out_dir / "pose.json")

            render_mask, render_edges, _ = render_points(
                pts,
                best_pose["yaw_deg"], best_pose["pitch_deg"], size,
                pad=0.12, point_radius=1,
                roll=best_pose["roll_deg"],
                scale_override=best_pose["scale_px_per_unit"],
                tx_px=best_pose["tx_px"], ty_px=best_pose["ty_px"],
                flip_x=best_pose.get("flip_x", False),
                flip_y=best_pose.get("flip_y", False),
            )

            overlay_path = img_out_dir / "overlay.png"
            overlay_m1(image_path, render_mask, render_edges,
                       img_mask, img_contour, overlay_path)
            overlay_paths.append(overlay_path)
            best_rows.append({"image": image_path.name, "group": gname, **best_pose})

    # Summary CSV
    csv_path = args.out / "best_poses.csv"
    if best_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
            writer.writeheader()
            writer.writerows(best_rows)
        print(f"\nWrote {csv_path}")

    # Contact sheet
    if overlay_paths:
        sheet_path = args.out / "contact_sheet.png"
        make_contact_sheet(overlay_paths, sheet_path)
        print(f"Wrote {sheet_path}")

    print("Done.")


if __name__ == "__main__":
    main()
