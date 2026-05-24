from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def load_xyz(path: Path, max_points: int | None, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns in {path}")

    points = data[:, :3]
    normals = data[:, 3:6] if data.shape[1] >= 6 else None

    if max_points and len(points) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        normals = normals[idx] if normals is not None else None

    points = points - points.mean(axis=0, keepdims=True)
    return points, normals


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])

    rz = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float32,
    )
    ry = np.array(
        [
            [math.cos(roll), 0.0, math.sin(roll)],
            [0.0, 1.0, 0.0],
            [-math.sin(roll), 0.0, math.cos(roll)],
        ],
        dtype=np.float32,
    )
    return ry @ rx @ rz


def dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant")
        out = (
            padded[:-2, :-2]
            | padded[:-2, 1:-1]
            | padded[:-2, 2:]
            | padded[1:-1, :-2]
            | padded[1:-1, 1:-1]
            | padded[1:-1, 2:]
            | padded[2:, :-2]
            | padded[2:, 1:-1]
            | padded[2:, 2:]
        )
    return out


def edge_from_mask(mask: np.ndarray) -> np.ndarray:
    eroded = mask & np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
    return mask & ~eroded


def image_edges(path: Path, size: tuple[int, int], threshold_percentile: float = 88.0) -> np.ndarray:
    img = Image.open(path).convert("L").resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(ImageOps.autocontrast(img), dtype=np.float32)

    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
    gy[1:-1, :] = arr[2:, :] - arr[:-2, :]
    mag = np.hypot(gx, gy)

    threshold = np.percentile(mag, threshold_percentile)
    edges = mag >= max(threshold, 12.0)
    return dilate(edges, 1)


def render_points(
    points: np.ndarray,
    yaw: float,
    pitch: float,
    size: tuple[int, int],
    pad: float,
    point_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    rotated = points @ rotation_matrix(yaw, pitch).T
    xy = rotated[:, :2]
    z = rotated[:, 2]

    span = np.maximum(np.ptp(xy, axis=0), 1e-6)
    scale = min((width * (1.0 - pad)) / span[0], (height * (1.0 - pad)) / span[1])
    px = ((xy[:, 0] - xy[:, 0].mean()) * scale + width / 2.0).astype(np.int32)
    py = (height / 2.0 - (xy[:, 1] - xy[:, 1].mean()) * scale).astype(np.int32)

    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px, py, z = px[valid], py[valid], z[valid]

    order = np.argsort(z)
    mask = np.zeros((height, width), dtype=bool)
    depth = np.zeros((height, width), dtype=np.float32)

    for x, y, zz in zip(px[order], py[order], z[order]):
        y0, y1 = max(0, y - point_radius), min(height, y + point_radius + 1)
        x0, x1 = max(0, x - point_radius), min(width, x + point_radius + 1)
        mask[y0:y1, x0:x1] = True
        depth[y0:y1, x0:x1] = zz

    mask = dilate(mask, 1)
    edges = dilate(edge_from_mask(mask), 1)
    return mask, edges


def score_edges(render_edges: np.ndarray, image_edges_mask: np.ndarray) -> float:
    if render_edges.sum() == 0 or image_edges_mask.sum() == 0:
        return 0.0

    image_near = dilate(image_edges_mask, 3)
    render_near = dilate(render_edges, 3)
    precision = (render_edges & image_near).sum() / render_edges.sum()
    recall = (image_edges_mask & render_near).sum() / image_edges_mask.sum()
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def infer_capture_group(path: Path) -> str:
    name = path.stem.lower()
    if "front" in name:
        return "top"
    if "back" in name:
        return "bottom"
    return "unknown"


def overlay_image(camera_path: Path, render_edges: np.ndarray, image_edges_mask: np.ndarray, out_path: Path) -> None:
    base = Image.open(camera_path).convert("RGB").resize((render_edges.shape[1], render_edges.shape[0]), Image.Resampling.LANCZOS)
    arr = np.asarray(base).copy()
    arr[image_edges_mask] = (40, 180, 255)
    arr[render_edges] = (255, 70, 50)
    overlap = image_edges_mask & render_edges
    arr[overlap] = (255, 230, 60)
    Image.fromarray(arr).save(out_path)


def make_contact_sheet(items: list[Path], out_path: Path, thumb_width: int = 384) -> None:
    thumbs = []
    for path in items:
        img = Image.open(path).convert("RGB")
        ratio = thumb_width / img.width
        thumb = img.resize((thumb_width, int(img.height * ratio)), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb.width, thumb.height + 24), "white")
        canvas.paste(thumb, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, thumb.height + 5), path.stem, fill=(20, 20, 20))
        thumbs.append(canvas)

    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    w = cols * thumb_width
    h = rows * max(t.height for t in thumbs)
    sheet = Image.new("RGB", (w, h), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * thumb_width, (i // cols) * thumb.height))
    sheet.save(out_path)


def parse_degrees(value: str) -> list[float]:
    if ":" in value:
        start, stop, step = map(float, value.split(":"))
        return list(np.arange(start, stop + step * 0.5, step))
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Match STEP-derived XYZ views to calibrated camera images.")
    parser.add_argument("--xyz", type=Path, default=Path("basic_shapes/s_rice.xyz"))
    parser.add_argument("--images", type=Path, default=Path("camera_images"))
    parser.add_argument("--out", type=Path, default=Path("outputs/xyz_view_matching"))
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=324)
    parser.add_argument("--yaw", default="0:345:15", help="Comma list or start:stop:step degrees")
    parser.add_argument("--pitch", default="-25,0,25", help="Comma list or start:stop:step degrees")
    parser.add_argument("--max-points", type=int, default=220_000)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    points, normals = load_xyz(args.xyz, args.max_points, args.seed)
    size = (args.width, args.height)
    yaw_values = parse_degrees(args.yaw)
    pitch_values = parse_degrees(args.pitch)
    image_paths = sorted(args.images.glob("*.jpg"))

    rows: list[dict[str, str | float]] = []
    best_overlays: list[Path] = []
    edge_cache = {path: image_edges(path, size) for path in image_paths}

    rendered_cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
    for yaw in yaw_values:
        for pitch in pitch_values:
            rendered_cache[(yaw, pitch)] = render_points(points, yaw, pitch, size, pad=0.12, point_radius=args.point_radius)

    for image_path in image_paths:
        image_edge = edge_cache[image_path]
        best = None
        for (yaw, pitch), (_, render_edge) in rendered_cache.items():
            match_score = score_edges(render_edge, image_edge)
            row = {
                "image": image_path.name,
                "group": infer_capture_group(image_path),
                "yaw_deg": yaw,
                "pitch_deg": pitch,
                "score": match_score,
            }
            rows.append(row)
            if best is None or match_score > best["score"]:
                best = row

        assert best is not None
        _, best_edges = rendered_cache[(float(best["yaw_deg"]), float(best["pitch_deg"]))]
        overlay_path = args.out / f"{image_path.stem}_best_yaw{int(best['yaw_deg'])}_pitch{int(best['pitch_deg'])}.png"
        overlay_image(image_path, best_edges, image_edge, overlay_path)
        best_overlays.append(overlay_path)

    with (args.out / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "group", "yaw_deg", "pitch_deg", "score"])
        writer.writeheader()
        writer.writerows(rows)

    best_rows = []
    for image_name in sorted({str(row["image"]) for row in rows}):
        image_rows = [row for row in rows if row["image"] == image_name]
        best_rows.append(max(image_rows, key=lambda row: float(row["score"])))

    with (args.out / "best_matches.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "group", "yaw_deg", "pitch_deg", "score"])
        writer.writeheader()
        writer.writerows(best_rows)

    group_rows = []
    for group in sorted({str(row["group"]) for row in rows}):
        group_candidates: dict[tuple[float, float], list[float]] = {}
        for row in rows:
            if row["group"] != group:
                continue
            key = (float(row["yaw_deg"]), float(row["pitch_deg"]))
            group_candidates.setdefault(key, []).append(float(row["score"]))
        for (yaw, pitch), scores in group_candidates.items():
            group_rows.append(
                {
                    "group": group,
                    "yaw_deg": yaw,
                    "pitch_deg": pitch,
                    "mean_score": float(np.mean(scores)),
                    "image_count": len(scores),
                }
            )
    group_rows.sort(key=lambda row: (str(row["group"]), -float(row["mean_score"])))
    with (args.out / "best_matches_by_group.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "yaw_deg", "pitch_deg", "mean_score", "image_count"])
        writer.writeheader()
        seen_groups = set()
        for row in group_rows:
            if row["group"] in seen_groups:
                continue
            seen_groups.add(row["group"])
            writer.writerow(row)

    make_contact_sheet(best_overlays, args.out / "best_overlays_contact_sheet.png")

    print(f"Loaded {len(points):,} points from {args.xyz}")
    print(f"Scored {len(image_paths)} images against {len(rendered_cache)} candidate views")
    print(f"Wrote {args.out / 'best_matches.csv'}")
    print(f"Wrote {args.out / 'best_matches_by_group.csv'}")
    print(f"Wrote {args.out / 'best_overlays_contact_sheet.png'}")


if __name__ == "__main__":
    main()
