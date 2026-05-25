"""Step-by-step feature extraction visualization from XYZ point cloud.

Shows how geometric features are extracted from the 3D data and what they
look like when projected to the top and bottom camera perspectives.

Outputs (outputs/feature_viz/):
  step1_raw.png          - raw point cloud: top / bottom orthographic views
  step2_normals.png      - surface normal direction map (|nx|R |ny|G |nz|B)
  step3_clusters.png     - normal direction clusters (same-orientation surfaces)
  step4_planes.png       - individual planes after distance splitting
  step5_boundaries.png   - inter-plane boundary edges
  step6_circles.png      - circular hole candidates (top & bottom plane projection)
  summary.png            - all steps side by side

Usage:
    python scripts/visualize_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from PIL import Image, ImageDraw

from matching3d2d.io.xyz import load_xyz
from matching3d2d.geometry.transforms import rotation_matrix
from matching3d2d.render.point_renderer import dilate, edge_from_mask

# ── Constants ──────────────────────────────────────────────────────────────────

PALETTE = np.array([
    [220,  60,  60],  # red
    [ 60, 200,  60],  # green
    [ 60, 110, 240],  # blue
    [240, 200,  40],  # yellow
    [190,  60, 220],  # purple
    [ 40, 200, 210],  # cyan
    [240, 140,  40],  # orange
    [160, 220,  60],  # lime
    [210, 110, 100],  # salmon
    [100, 180, 210],  # sky
    [240, 100, 160],  # pink
    [130, 220, 160],  # mint
], dtype=np.uint8)

BG = (18, 18, 18)
SIZE = (576, 324)
PAD = 0.12
POINT_R = 1

# Camera-representative views
VIEWS = {
    "top":    dict(yaw=0, pitch=88),   # nearly straight down
    "bottom": dict(yaw=0, pitch=-88),  # nearly straight up
}
VIEW_ORDER = ["top", "bottom"]


# ── Projection & rendering ─────────────────────────────────────────────────────

def project(points: np.ndarray, yaw: float, pitch: float,
            size: tuple[int, int] = SIZE, pad: float = PAD):
    w, h = size
    R = rotation_matrix(yaw, pitch, 0.0)
    rot = points @ R.T
    xy, z = rot[:, :2], rot[:, 2]
    span = np.maximum(np.ptp(xy, axis=0), 1e-6)
    scale = float(min((w * (1 - pad)) / span[0], (h * (1 - pad)) / span[1]))
    px = np.round((xy[:, 0] - xy[:, 0].mean()) * scale + w / 2).astype(np.int32)
    py = np.round(h / 2 - (xy[:, 1] - xy[:, 1].mean()) * scale).astype(np.int32)
    return px, py, z, scale


def render_colored(points: np.ndarray, colors: np.ndarray,
                   yaw: float, pitch: float,
                   size: tuple[int, int] = SIZE) -> np.ndarray:
    """Orthographic render with per-point RGB colors (painter's algorithm)."""
    w, h = size
    px, py, z, _ = project(points, yaw, pitch, size)
    valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    pxv, pyv, zv = px[valid], py[valid], z[valid]
    cv = colors[valid]
    order = np.argsort(zv)          # back→front, front overwrites
    img = np.full((h, w, 3), BG, dtype=np.uint8)
    img[pyv[order], pxv[order]] = cv[order]
    return img


def render_silhouette(points: np.ndarray, yaw: float, pitch: float,
                      size: tuple[int, int] = SIZE) -> np.ndarray:
    """Binary silhouette mask."""
    w, h = size
    px, py, z, _ = project(points, yaw, pitch, size)
    valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    mask = np.zeros((h, w), dtype=bool)
    if valid.any():
        mask[py[valid], px[valid]] = True
    mask = dilate(mask, POINT_R)
    return dilate(mask, 1)


def silhouette_to_rgb(mask: np.ndarray,
                      fg=(180, 180, 180), bg=BG) -> np.ndarray:
    img = np.full((*mask.shape, 3), bg, dtype=np.uint8)
    img[mask] = fg
    return img


def colored_to_pil(arr: np.ndarray, label: str = "",
                   label_color=(220, 220, 100)) -> Image.Image:
    img = Image.fromarray(arr.astype(np.uint8))
    if label:
        draw = ImageDraw.Draw(img)
        draw.text((6, 5), label, fill=label_color)
    return img


# ── Normal clustering ──────────────────────────────────────────────────────────

def normalize_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def kmeans_sphere(normals_norm: np.ndarray, k: int,
                  n_iter: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """K-means clustering on the unit sphere (sign-invariant cosine distance)."""
    rng = np.random.default_rng(seed)
    centers = normals_norm[rng.choice(len(normals_norm), k, replace=False)].copy()
    labels = np.zeros(len(normals_norm), dtype=np.int32)

    for _ in range(n_iter):
        sim = np.abs(normals_norm @ centers.T)   # (N, k)
        labels = np.argmax(sim, axis=1)
        for c in range(k):
            pts = normals_norm[labels == c]
            if len(pts) == 0:
                continue
            mean = pts.mean(axis=0)
            nm = np.linalg.norm(mean)
            centers[c] = mean / max(nm, 1e-8)

    # Sort clusters by size (largest first)
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    remap = np.empty(k, dtype=np.int32)
    remap[order] = np.arange(k)
    labels = remap[labels]
    centers = centers[order]
    return labels, centers


# ── Plane splitting ────────────────────────────────────────────────────────────

def split_into_planes(points: np.ndarray, cluster_indices: np.ndarray,
                      normal: np.ndarray,
                      dist_thresh: float = 0.3,
                      min_pts: int = 300) -> list[np.ndarray]:
    """Split a normal cluster into individual planes by distance histogram."""
    pts = points[cluster_indices]
    n_hat = normal / np.linalg.norm(normal)
    dists = pts @ n_hat

    span = dists.max() - dists.min()
    if span < dist_thresh * 2:
        return [cluster_indices]

    n_bins = max(20, int(span / (dist_thresh * 0.5)))
    hist, edges = np.histogram(dists, bins=n_bins)

    # Peak detection: local maxima above 5% of max count
    thresh = max(min_pts, hist.max() * 0.05)
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] >= thresh and hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]:
            peaks.append((edges[i] + edges[i + 1]) / 2)

    if not peaks:
        return [cluster_indices]

    groups = []
    for peak_d in peaks:
        mask = np.abs(dists - peak_d) < dist_thresh
        if mask.sum() >= min_pts:
            groups.append(cluster_indices[mask])
    return groups if groups else [cluster_indices]


# ── Boundary detection ─────────────────────────────────────────────────────────

def label_image(points: np.ndarray, plane_labels: np.ndarray,
                yaw: float, pitch: float,
                size: tuple[int, int] = SIZE) -> np.ndarray:
    """Render integer plane-label map (−1 = background)."""
    w, h = size
    px, py, z, _ = project(points, yaw, pitch, size)
    valid = (px >= 0) & (px < w) & (py >= 0) & (py < h) & (plane_labels >= 0)
    limg = np.full((h, w), -1, dtype=np.int32)
    pxv, pyv, zv = px[valid], py[valid], z[valid]
    lv = plane_labels[valid]
    order = np.argsort(zv)            # back→front
    limg[pyv[order], pxv[order]] = lv[order]
    return limg


def boundary_mask(limg: np.ndarray) -> np.ndarray:
    """Pixels whose 4-neighbors include a different valid label."""
    valid = limg >= 0
    bnd = np.zeros_like(valid)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted_l = np.roll(np.roll(limg, dr, 0), dc, 1)
        shifted_v = np.roll(np.roll(valid, dr, 0), dc, 1)
        bnd |= valid & shifted_v & (limg != shifted_l)
    return bnd


# ── Circle / hole detection ────────────────────────────────────────────────────

def detect_holes(mask: np.ndarray, min_hole_px: int = 30) -> list[tuple[int, int, int]]:
    """Find circular holes (empty connected regions inside the object mask).

    Returns list of (cy, cx, radius) in pixels.
    """
    from collections import deque

    # Interior = pixels inside bounding box of mask but NOT in mask
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return []
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])

    interior_bg = ~mask
    interior_bg[:r0, :] = False
    interior_bg[r1 + 1:, :] = False
    interior_bg[:, :c0] = False
    interior_bg[:, c1 + 1:] = False

    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    holes = []

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if not interior_bg[r, c] or visited[r, c]:
                continue
            # BFS
            q: deque[tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            region: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                region.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and interior_bg[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            if len(region) >= min_hole_px:
                ys = [p[0] for p in region]
                xs = [p[1] for p in region]
                cy = int(np.mean(ys))
                cx = int(np.mean(xs))
                r_est = int(np.sqrt(len(region) / np.pi))
                holes.append((cy, cx, max(r_est, 3)))

    return holes


# ── Panel assembly ─────────────────────────────────────────────────────────────

def make_row(panels: list[np.ndarray], row_label: str = "",
             label_h: int = 22) -> np.ndarray:
    """Stack panels horizontally and add a row label bar."""
    row = np.concatenate(panels, axis=1)
    if row_label:
        bar = np.full((label_h, row.shape[1], 3), (30, 30, 30), dtype=np.uint8)
        img = np.concatenate([bar, row], axis=0)
        pil = Image.fromarray(img)
        ImageDraw.Draw(pil).text((6, 4), row_label, fill=(200, 200, 80))
        return np.asarray(pil)
    return row


def save_row(panels: list[np.ndarray], labels: list[str],
             row_label: str, out_path: Path) -> np.ndarray:
    pil_panels = [colored_to_pil(p, l) for p, l in zip(panels, labels)]
    widths = [p.width for p in pil_panels]
    h = max(p.height for p in pil_panels)
    strip = Image.new("RGB", (sum(widths), h), BG)
    x = 0
    for p in pil_panels:
        strip.paste(p, (x, 0))
        x += p.width
    row_arr = np.asarray(strip)
    result = make_row([row_arr], row_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(out_path)
    print(f"  Saved {out_path.name}")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    xyz_path = Path("basic_shapes/s_rice.xyz")
    out_dir = Path("outputs/feature_viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {xyz_path} ...")
    points, normals = load_xyz(xyz_path, max_points=None, seed=0)
    print(f"  {len(points):,} points, normals={'yes' if normals is not None else 'no'}")

    normals_norm = normalize_rows(normals) if normals is not None else None

    summary_rows: list[np.ndarray] = []

    # ── Step 1: Raw point cloud ─────────────────────────────────────────────
    print("\nStep 1: Raw XYZ views")
    raw_panels, raw_labels = [], []
    for vname in VIEW_ORDER:
        v = VIEWS[vname]
        sil = render_silhouette(points, v["yaw"], v["pitch"])
        rgb = silhouette_to_rgb(sil, fg=(160, 160, 180))
        raw_panels.append(rgb)
        raw_labels.append(f"{vname}  yaw={v['yaw']} pitch={v['pitch']}")
    row = save_row(raw_panels, raw_labels, "Step 1 | Raw XYZ point cloud", out_dir / "step1_raw.png")
    summary_rows.append(row)

    # ── Step 2: Normal direction map ────────────────────────────────────────
    print("\nStep 2: Normal direction map  (R=|nx|  G=|ny|  B=|nz|)")
    if normals_norm is not None:
        normal_colors = (np.abs(normals_norm) * 255).astype(np.uint8)
        norm_panels, norm_labels = [], []
        for vname in VIEW_ORDER:
            v = VIEWS[vname]
            rgb = render_colored(points, normal_colors, v["yaw"], v["pitch"])
            norm_panels.append(rgb)
            norm_labels.append(f"{vname}  |nx|→R  |ny|→G  |nz|→B")
        row = save_row(norm_panels, norm_labels, "Step 2 | Normal direction map", out_dir / "step2_normals.png")
        summary_rows.append(row)
    else:
        print("  (skipped — no normals in XYZ file)")

    # ── Step 3: Normal clustering ───────────────────────────────────────────
    print("\nStep 3: Normal clustering  (k=8 k-means on unit sphere)")
    K = 8
    if normals_norm is not None:
        cluster_labels, cluster_centers = kmeans_sphere(normals_norm, k=K, seed=7)
        point_colors_cluster = PALETTE[cluster_labels % len(PALETTE)]

        clust_panels, clust_labels = [], []
        for vname in VIEW_ORDER:
            v = VIEWS[vname]
            rgb = render_colored(points, point_colors_cluster, v["yaw"], v["pitch"])
            clust_panels.append(rgb)
            clust_labels.append(f"{vname}  {K} normal-direction clusters")
        row = save_row(clust_panels, clust_labels,
                       f"Step 3 | Normal clusters (k={K})  each color = one surface orientation",
                       out_dir / "step3_clusters.png")
        summary_rows.append(row)

        # Print cluster summary
        for cid in range(K):
            mask = cluster_labels == cid
            if mask.sum() < 100:
                continue
            cn = cluster_centers[cid]
            dominant = ["±X", "±Y", "±Z"][np.argmax(np.abs(cn))]
            print(f"    Cluster {cid}: {mask.sum():6,} pts  dominant={dominant}  "
                  f"n=[{cn[0]:+.2f} {cn[1]:+.2f} {cn[2]:+.2f}]")
    else:
        cluster_labels = np.zeros(len(points), dtype=np.int32)
        cluster_centers = np.array([[0.0, 0.0, 1.0]])

    # ── Step 4: Individual planes ───────────────────────────────────────────
    print("\nStep 4: Individual plane extraction  (distance histogram split)")
    plane_labels = np.full(len(points), -1, dtype=np.int32)
    plane_id = 0
    plane_info = []

    if normals_norm is not None:
        for cid in range(K):
            cidx = np.where(cluster_labels == cid)[0]
            if len(cidx) < 300:
                continue
            cn = cluster_centers[cid]
            sub_planes = split_into_planes(points, cidx, cn,
                                           dist_thresh=0.3, min_pts=300)
            for sp in sub_planes:
                if len(sp) < 300:
                    continue
                plane_labels[sp] = plane_id
                dominant = ["X", "Y", "Z"][np.argmax(np.abs(cn))]
                plane_info.append(dict(id=plane_id, cluster=cid,
                                       n_pts=len(sp), dominant_axis=dominant,
                                       normal=cn.tolist()))
                plane_id += 1
    else:
        # Fallback: single plane
        plane_labels[:] = 0
        plane_info = [dict(id=0, cluster=0, n_pts=len(points),
                           dominant_axis="Z", normal=[0, 0, 1])]

    point_colors_plane = np.full((len(points), 3), 40, dtype=np.uint8)
    for info in plane_info:
        pid = info["id"]
        mask = plane_labels == pid
        point_colors_plane[mask] = PALETTE[pid % len(PALETTE)]

    plane_panels, plane_labels_list = [], []
    for vname in VIEW_ORDER:
        v = VIEWS[vname]
        rgb = render_colored(points, point_colors_plane, v["yaw"], v["pitch"])
        plane_panels.append(rgb)
        plane_labels_list.append(f"{vname}  {plane_id} planes  (gray=unassigned)")
    row = save_row(plane_panels, plane_labels_list,
                   f"Step 4 | Individual planes  ({plane_id} total, each color = one plane)",
                   out_dir / "step4_planes.png")
    summary_rows.append(row)
    for info in plane_info:
        print(f"    Plane {info['id']:2d}: {info['n_pts']:6,} pts  "
              f"cluster={info['cluster']}  dominant={info['dominant_axis']}")

    # ── Step 5: Plane boundaries ────────────────────────────────────────────
    print("\nStep 5: Inter-plane boundary edges")
    bnd_panels, bnd_labels_list = [], []
    for vname in VIEW_ORDER:
        v = VIEWS[vname]
        limg = label_image(points, plane_labels, v["yaw"], v["pitch"])
        bnd = boundary_mask(limg)
        bnd_dil = dilate(bnd, 1)

        # Base: dimmed plane-colored image
        base = render_colored(points, point_colors_plane, v["yaw"], v["pitch"])
        base = (base * 0.4).astype(np.uint8)
        base[bnd_dil] = (255, 220, 60)   # yellow boundary

        bnd_panels.append(base)
        bnd_labels_list.append(f"{vname}  yellow = plane boundary edges")
    row = save_row(bnd_panels, bnd_labels_list,
                   "Step 5 | Plane boundary edges  (yellow = transition between planes)",
                   out_dir / "step5_boundaries.png")
    summary_rows.append(row)

    # ── Step 6: Circular hole detection (top & bottom) ──────────────────────
    print("\nStep 6: Circular hole candidates  (top & bottom views)")
    # Identify top-facing and bottom-facing planes (dominant Z axis)
    z_planes_up = [p for p in plane_info
                   if p["dominant_axis"] == "Z" and p["normal"][2] > 0]
    z_planes_dn = [p for p in plane_info
                   if p["dominant_axis"] == "Z" and p["normal"][2] < 0]

    top_idx = np.concatenate(
        [np.where(plane_labels == p["id"])[0] for p in z_planes_up]
    ) if z_planes_up else np.arange(len(points))
    bot_idx = np.concatenate(
        [np.where(plane_labels == p["id"])[0] for p in z_planes_dn]
    ) if z_planes_dn else np.arange(len(points))

    circ_panels, circ_labels_list = [], []
    for vname, idx_set, facing in [
        ("top",    top_idx, "top-facing (nz>0)"),
        ("bottom", bot_idx, "bottom-facing (nz<0)"),
    ]:
        v = VIEWS[vname]
        pts_sub = points[idx_set] if len(idx_set) > 0 else points

        sil = render_silhouette(pts_sub, v["yaw"], v["pitch"])
        base_gray = silhouette_to_rgb(sil, fg=(80, 120, 180))

        holes = detect_holes(sil, min_hole_px=20)
        pil_img = Image.fromarray(base_gray)
        draw = ImageDraw.Draw(pil_img)
        for cy, cx, r in holes:
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)],
                         outline=(255, 80, 80), width=2)
            draw.ellipse([(cx - 2, cy - 2), (cx + 2, cy + 2)],
                         fill=(255, 80, 80))
        circ_panels.append(np.asarray(pil_img))
        circ_labels_list.append(
            f"{vname}  {facing}  {len(holes)} hole candidates  (red circles)"
        )
        print(f"    {vname}: {len(holes)} hole candidates  "
              f"(from {len(z_planes_up if vname == 'top' else z_planes_dn)} planes, "
              f"{len(idx_set):,} pts)")

    row = save_row(circ_panels, circ_labels_list,
                   "Step 6 | Circular hole candidates  (red = detected holes/cutouts)",
                   out_dir / "step6_circles.png")
    summary_rows.append(row)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\nBuilding summary sheet ...")
    max_w = max(r.shape[1] for r in summary_rows)
    padded = []
    for r in summary_rows:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1], 3), 15, dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)
    summary = np.concatenate(padded, axis=0)
    summary_path = out_dir / "summary.png"
    Image.fromarray(summary).save(summary_path)
    print(f"  Saved summary.png  ({summary.shape[1]}x{summary.shape[0]} px)")

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
