"""M1: Object-mask-aware coarse search and 2D refinement.

Usage:
    python scripts/align_camera_images.py
    python scripts/align_camera_images.py --no-refine          # skip 2D refinement (fast)
    python scripts/align_camera_images.py --top-k 1            # refine only best coarse candidate
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow importing the package without installation.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from matching3d2d.io.xyz import load_xyz
from matching3d2d.io.camera_images import load_image_paths, infer_capture_group
from matching3d2d.render.point_renderer import render_points
from matching3d2d.vision.edges import image_edges
from matching3d2d.vision.masks import extract_object_mask
from matching3d2d.vision.contours import external_contour, internal_edges
from matching3d2d.align.coarse import run_coarse_search
from matching3d2d.align.refine2d import refine2d
from matching3d2d.viz.overlays import overlay_m1, make_contact_sheet
from matching3d2d.viz.reports import save_pose_json


def parse_degrees(value: str) -> list[float]:
    if ":" in value:
        start, stop, step = map(float, value.split(":"))
        return list(np.arange(start, stop + step * 0.5, step))
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M1: mask-aware coarse search + 2D refinement for XYZ-to-camera alignment."
    )
    parser.add_argument("--xyz", type=Path, default=Path("basic_shapes/s_rice.xyz"))
    parser.add_argument("--images", type=Path, default=Path("camera_images"))
    parser.add_argument("--out", type=Path, default=Path("outputs/alignments"))
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=324)
    parser.add_argument("--yaw", default="0:345:15", help="start:stop:step or comma list (degrees)")
    parser.add_argument("--pitch", default="-25,0,25", help="comma list or start:stop:step (degrees)")
    parser.add_argument("--max-points", type=int, default=220_000)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3, help="Number of coarse candidates to refine per image")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-refine", action="store_true", help="Skip 2D refinement; use coarse best pose")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    size = (args.width, args.height)
    yaw_values = parse_degrees(args.yaw)
    pitch_values = parse_degrees(args.pitch)

    print(f"Loading XYZ from {args.xyz} ...")
    points, _ = load_xyz(args.xyz, args.max_points, args.seed)
    print(f"  {len(points):,} points loaded")

    image_paths = load_image_paths(args.images)
    print(f"Processing {len(image_paths)} images | {len(yaw_values)} yaw x {len(pitch_values)} pitch candidates")
    if not args.no_refine:
        n_refine = 7 * 5 * 5 * 5  # scale_deltas * tx * ty * roll
        print(f"  Refinement: top-{args.top_k} x {n_refine} steps per image (use --no-refine to skip)")

    best_rows: list[dict] = []
    overlay_paths: list[Path] = []

    for image_path in image_paths:
        group = infer_capture_group(image_path)
        print(f"\n  [{group}] {image_path.name}")

        # --- 2D image features ---
        img_all_edges = image_edges(image_path, size)
        img_mask = extract_object_mask(image_path, size)
        img_contour = external_contour(img_mask)
        img_internal = internal_edges(img_all_edges, img_mask, img_contour)

        mask_coverage = float(img_mask.sum()) / float(img_mask.size)
        print(f"    Object mask coverage: {mask_coverage:.1%}")

        # --- Coarse search ---
        coarse = run_coarse_search(
            points,
            img_mask,
            img_contour,
            img_internal,
            yaw_values,
            pitch_values,
            size,
            point_radius=args.point_radius,
        )
        best_coarse = coarse[0]
        print(
            f"    Coarse best: yaw={best_coarse['yaw_deg']:.0f}  "
            f"pitch={best_coarse['pitch_deg']:.0f}  "
            f"score={best_coarse['score']:.4f}"
        )

        # --- 2D refinement ---
        if args.no_refine:
            best_pose = best_coarse
        else:
            top_k = coarse[: args.top_k]
            refined = [
                refine2d(
                    points,
                    p,
                    img_mask,
                    img_contour,
                    img_internal,
                    size,
                    point_radius=args.point_radius,
                )
                for p in top_k
            ]
            best_pose = max(refined, key=lambda r: r["score"])
            print(
                f"    Refined:    yaw={best_pose['yaw_deg']:.0f}  "
                f"pitch={best_pose['pitch_deg']:.0f}  "
                f"roll={best_pose['roll_deg']:.1f}  "
                f"tx={best_pose['tx_px']:.0f}  ty={best_pose['ty_px']:.0f}  "
                f"score={best_pose['score']:.4f}"
            )

        # --- Save outputs ---
        img_out_dir = args.out / image_path.stem
        img_out_dir.mkdir(parents=True, exist_ok=True)

        save_pose_json(best_pose, img_out_dir / "pose.json")

        render_mask, render_edges, _ = render_points(
            points,
            best_pose["yaw_deg"],
            best_pose["pitch_deg"],
            size,
            pad=0.12,
            point_radius=args.point_radius,
            roll=best_pose["roll_deg"],
            scale_override=best_pose["scale_px_per_unit"],
            tx_px=best_pose["tx_px"],
            ty_px=best_pose["ty_px"],
        )

        overlay_path = img_out_dir / "overlay_geometry.png"
        overlay_m1(image_path, render_mask, render_edges, img_mask, img_contour, overlay_path)
        overlay_paths.append(overlay_path)

        best_rows.append({"image": image_path.name, "group": group, **best_pose})

    # --- Summary CSV ---
    csv_path = args.out / "best_poses.csv"
    if best_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
            writer.writeheader()
            writer.writerows(best_rows)
        print(f"\nWrote {csv_path}")

    # --- Contact sheet ---
    if overlay_paths:
        sheet_path = args.out / "best_overlays_contact_sheet.png"
        make_contact_sheet(overlay_paths, sheet_path)
        print(f"Wrote {sheet_path}")

    print("Done.")


if __name__ == "__main__":
    main()
