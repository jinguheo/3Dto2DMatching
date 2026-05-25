# 3Dto2DMatching

Pipeline experiments for matching STEP-derived 3D geometry to calibrated 2D camera images.

## Data

- `basic_shapes/s_rice.stp`: source STEP design file.
- `basic_shapes/s_rice.xyz`: dense point and normal samples extracted from the STEP file.
- `camera_images/`: calibrated top/bottom camera images.

## M1: Mask-Aware Alignment (current)

Run the M1 pipeline — object mask extraction, contour-based scoring, and 2D refinement:

```powershell
python scripts\align_camera_images.py
```

Options:

```powershell
# Fast mode: skip 2D refinement, use coarse best pose only
python scripts\align_camera_images.py --no-refine

# Refine only the single best coarse candidate (faster than default top-3)
python scripts\align_camera_images.py --top-k 1
```

Outputs written under `outputs/alignments/`:

- `best_poses.csv` — best pose per image
- `best_overlays_contact_sheet.png` — all 6 overlay previews
- `<image_stem>/pose.json` — weak-perspective pose (yaw, pitch, roll, scale, tx, ty, score)
- `<image_stem>/overlay_geometry.png` — camera image with CAD edges and object contour overlaid

## M0: Baseline View Search

Original edge-based view retrieval (no mask, no refinement):

```powershell
python scripts\match_xyz_views.py
```

Outputs written under `outputs/xyz_view_matching/`.

## Design

See:

- `docs/alignment_system_design.md` for the overall primitive extraction, geometry alignment, color alignment, shading alignment, and pose optimization pipeline.
- `docs/implementation_design.md` for concrete modules, schemas, rendering outputs, losses, and implementation milestones.
