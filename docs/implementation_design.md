# Implementation Design

This document turns the alignment concept into concrete modules, file formats, and implementation steps.

The project goal is to align CAD-derived geometry, color, and shading information to real camera observations. Geometry is the hard constraint. Color and shading are appearance cues that become useful once projected geometry is already close.

## Package Layout

Target source layout:

```text
src/
  matching3d2d/
    __init__.py
    config.py
    io/
      xyz.py
      camera_images.py
      step.py
      schema.py
    geometry/
      transforms.py
      primitives.py
      primitive_fit.py
      primitive_graph.py
    vision/
      masks.py
      edges.py
      contours.py
      color.py
      shading.py
    render/
      point_renderer.py
      primitive_renderer.py
      appearance_renderer.py
    align/
      coarse.py
      refine2d.py
      pose.py
      losses.py
      lighting.py
    viz/
      overlays.py
      reports.py
scripts/
  match_xyz_views.py
  extract_xyz_primitives.py
  align_camera_images.py
configs/
  rice_default.yaml
```

Current baseline code remains in `scripts/match_xyz_views.py` until the package is introduced.

## Data Products

Generated files should stay under `outputs/` and remain untracked by git.

```text
outputs/
  xyz_view_matching/
    scores.csv
    best_matches.csv
    best_matches_by_group.csv
    best_overlays_contact_sheet.png
  primitives/
    xyz_primitives.json
    step_primitives.json
    primitive_graph.json
    primitive_preview.png
  alignments/
    camera_front_rice1076calibrated/
      pose.json
      primitive_matches.csv
      overlay_geometry.png
      overlay_color.png
      report.json
```

## Core Schemas

### Camera Image Record

```json
{
  "image_id": "camera_front_rice1076calibrated",
  "path": "camera_images/camera_front_rice1076calibrated.jpg",
  "group": "top",
  "width": 1152,
  "height": 648,
  "intrinsics": null,
  "distortion": null,
  "is_distortion_corrected": true
}
```

When calibration data becomes available, `intrinsics` should use:

```json
{
  "fx": 1000.0,
  "fy": 1000.0,
  "cx": 576.0,
  "cy": 324.0
}
```

### 3D Primitive

Use one schema for STEP-derived exact primitives and XYZ-fitted primitives.

```json
{
  "id": "plane_0001",
  "type": "plane",
  "source": "xyz",
  "confidence": 0.94,
  "params": {
    "origin": [0.0, 0.0, 0.0],
    "normal": [0.0, 0.0, 1.0]
  },
  "support": {
    "point_count": 12345,
    "rms_error": 0.03,
    "indices_path": null
  },
  "bounds": {
    "points_3d": [[-10.0, -5.0, 0.0], [10.0, -5.0, 0.0]],
    "aabb": {
      "min": [-10.0, -5.0, -0.02],
      "max": [10.0, 5.0, 0.02]
    }
  },
  "appearance": {
    "base_color_rgb": null,
    "material": null,
    "roughness": null,
    "metallic": 0.0,
    "shading_model": "lambertian"
  }
}
```

Primitive types:

- `plane`
- `line`
- `circle`
- `cylinder`
- `box`
- `slot`
- `freeform`

### Pose

Before camera intrinsics are known, store weak-perspective pose:

```json
{
  "model": "weak_perspective",
  "yaw_deg": 105.0,
  "pitch_deg": 0.0,
  "roll_deg": 0.0,
  "scale_px_per_unit": 4.1,
  "tx_px": 0.0,
  "ty_px": 0.0,
  "score": 0.27
}
```

After intrinsics are known:

```json
{
  "model": "pinhole",
  "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  "translation": [0.0, 0.0, 1000.0],
  "intrinsics": {
    "fx": 1000.0,
    "fy": 1000.0,
    "cx": 576.0,
    "cy": 324.0
  },
  "score": 0.86
}
```

## Primitive Extraction Design

### XYZ Plane Extraction

Input:

- Dense `N x 3` points.
- Optional `N x 3` normals.

Steps:

1. Downsample points using voxel grid.
2. Cluster normals on the unit sphere.
3. For each dominant normal cluster, run seeded plane fitting.
4. Grow planar regions by point-to-plane distance and normal angle.
5. Remove small regions.
6. Estimate 2D plane boundary by projecting support points to local plane coordinates.
7. Simplify boundary into line segments and rectangle candidates.

Initial thresholds:

```text
voxel_size_mm = 0.5 to 1.0
normal_angle_deg = 8
plane_distance_mm = 0.15 to 0.3
min_plane_points = 300
boundary_simplify_mm = 0.5
```

### XYZ Circle/Hole Extraction

First version:

1. Use planar primitive supports.
2. Project each plane's points into local 2D coordinates.
3. Detect empty connected regions inside dense support masks.
4. Fit circles/ellipses to hole boundaries.
5. Lift circle centers and radii back to 3D plane coordinates.

This is more stable than looking for cylinders first, because camera images show holes as 2D circles/ellipses and the object appears mostly planar.

### STEP Primitive Extraction

Use OpenCascade/OCP when available.

Expected functions:

- `load_step_shape(path)`
- `iter_faces(shape)`
- `classify_face_surface(face)`
- `iter_edges(face)`
- `classify_edge_curve(edge)`
- `export_step_primitives(shape)`

STEP primitives should override XYZ primitives when they agree spatially. XYZ primitives remain useful for rendering density, noisy tolerance checks, and fallback.

## Rendering Design

### Point Rendering

Use for coarse retrieval.

- Orthographic projection while intrinsics are unknown.
- Dense point splatting with z-order.
- Outputs:
  - silhouette mask
  - edge mask
  - optional depth map

### Primitive Rendering

Use for refinement.

- Render projected primitive boundaries as ID-labeled masks.
- Preserve primitive IDs in a separate image-sized integer buffer.
- Render types:
  - plane boundary
  - line segment
  - circle/ellipse
  - cylinder silhouette if available

Outputs:

```text
render_edges: bool[H, W]
render_silhouette: bool[H, W]
primitive_id_map: int[H, W]
depth: float[H, W]
```

### Appearance Rendering

Use after a pose hypothesis exists.

Inputs:

- primitive geometry
- primitive base color/material if available from STEP or side metadata
- pose
- camera intrinsics or weak-perspective camera
- simple lighting parameters

Outputs:

```text
render_rgb: uint8[H, W, 3]
render_normal: float[H, W, 3]
render_albedo: uint8[H, W, 3]
render_shading: float[H, W]
primitive_id_map: int[H, W]
```

First implementation should use a simple Lambertian approximation:

```text
I = exposure * base_color * (ambient + max(0, dot(normal, light_dir)) * diffuse)
```

This is intentionally simple. The goal is not photorealistic rendering at first; it is to provide a stable appearance consistency signal for matching.

## 2D Image Feature Design

### Object Mask

The immediate improvement over baseline is to suppress background and internal texture noise.

Initial method:

1. Convert image to grayscale.
2. Estimate background from image borders.
3. Threshold absolute difference from background.
4. Morphologically close gaps.
5. Keep largest connected component.
6. Extract external contour.

Outputs:

- `object_mask`
- `external_contour`
- `masked_edges`

### Geometry Features

Extract:

- exterior contour edges
- internal high-confidence straight lines
- circle/ellipse candidates
- corners

Use exterior contour for initial alignment, then add internal features after pose is close.

### Color Features

Color must be used late, after geometry is close.

Per projected primitive:

- compute median RGB
- compute interquartile RGB range
- compare against expected color or neighboring-view consistency

Avoid raw pixel loss at first because lighting, exposure, and specular reflections can dominate.

### Shading Features

Shading should use low-frequency intensity structure rather than exact brightness.

For each aligned primitive region:

- convert the camera crop to normalized grayscale or Lab-L
- remove per-region median intensity
- compare blurred gradients or normalized intensity residuals
- ignore saturated pixels and strong specular highlights

Useful outputs:

- `image_luma_normalized`
- `image_luma_blurred`
- `image_shading_gradient`
- saturated/specular mask

## Alignment Algorithm

### Coarse Search

Search parameters:

```text
yaw: 0 to 345 step 15 deg
pitch: top/bottom prior dependent
roll: initially 0, later -10 to 10 deg
scale: fit-to-image bounding box
translation: centered
```

Score:

```text
score = 0.45 * contour_f1
      + 0.35 * silhouette_iou
      + 0.15 * internal_edge_f1
      + 0.05 * appearance_hint
```

Current baseline only approximates `internal_edge_f1`; M1 should add `contour_f1` and `silhouette_iou`. `appearance_hint` should remain zero until projected geometry is stable.

### 2D Refinement

For top-K coarse poses:

1. Optimize `scale`, `tx`, `ty`, and `roll`.
2. Use grid search first for robustness.
3. Follow with coordinate descent.
4. Re-render after each pose update.

Initial search:

```text
scale_delta: [-12%, -8%, -4%, 0, 4%, 8%, 12%]
tx_px: [-80, -40, 0, 40, 80]
ty_px: [-80, -40, 0, 40, 80]
roll_deg: [-8, -4, 0, 4, 8]
```

### Primitive Correspondence

After image-plane refinement:

1. Project every 3D primitive.
2. Match projected lines to 2D line segments by distance, angle, and overlap.
3. Match projected circles to 2D ellipses by center distance and radius/axis agreement.
4. Reject geometrically impossible matches.
5. Use accepted matches for pose optimization.

### Full Pose Optimization

Requires camera intrinsics for best results.

Loss terms:

- contour Chamfer distance
- primitive-to-feature distance
- silhouette IoU
- top/bottom pose prior
- color residual after convergence
- shading residual after geometry and color are stable

The optimizer should report failure when correspondences are ambiguous rather than forcing a pose.

### Color-Aware Alignment

After geometry refinement:

1. Project primitives to image regions.
2. Compute per-primitive median color in the real image.
3. Compare with CAD/material base color if available.
4. If no CAD color exists, learn per-primitive observed color as metadata for consistency across views.

Color score:

```text
L_color = robust_l1(normalized_image_color - rendered_or_expected_color)
```

Use this mostly to disambiguate geometrically similar poses or primitive matches.

### Shading-Aware Alignment

After geometry and rough color alignment:

1. Render primitive normals and visibility.
2. Estimate image-level lighting:
   - ambient
   - directional light vector
   - diffuse strength
   - exposure
3. Compare low-frequency rendered shading to observed luma.
4. Add the residual as a weak alignment term.

Shading score:

```text
L_shading = robust_l1(blur(norm_luma_image) - blur(norm_luma_render))
```

Use shading carefully:

- strong cue for face orientation once pose is close
- weak cue for initial retrieval
- unreliable around reflections, occlusions, overexposure, and material changes

## Top/Bottom View Handling

The six images should be represented as two capture groups.

Top and bottom groups can share:

- same CAD model
- same primitive set
- same camera intrinsics if the same camera/lens was used

They differ by:

- expected viewing direction
- visible primitive subset
- occlusion ordering
- expected color/lighting statistics

If `front`/`back` naming does not exactly match top/bottom, update one mapping function instead of hardcoding the assumption throughout the code.

## Quality Gates

For each milestone, require:

- A reproducible command in `README.md`.
- A generated visual contact sheet.
- CSV/JSON machine-readable outputs.
- At least one sanity check:
  - non-empty masks
  - non-empty primitive list
  - finite pose values
  - overlay dimensions match camera dimensions

## Next Code Milestone

Implement M1 in this order:

1. Move common XYZ/image/render functions from `scripts/match_xyz_views.py` into package modules.
2. Add object mask and external contour extraction.
3. Update matching score to include contour F1 and silhouette IoU.
4. Add 2D scale/translation/roll refinement around the best yaw/pitch.
5. Save `pose.json` and cleaner overlays per camera image.

This makes the baseline useful enough to guide primitive extraction without being dominated by texture edges.

After M1, add an appearance milestone:

1. Estimate object-region median colors in each camera image.
2. Render per-primitive normal maps from XYZ/primitive geometry.
3. Add a simple lighting fitter for an already aligned pose.
4. Save `overlay_color.png`, `overlay_shading.png`, and appearance residual maps.
