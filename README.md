# 3Dto2DMatching

Pipeline experiments for matching STEP-derived 3D geometry to calibrated 2D camera images.

## Data

- `basic_shapes/s_rice.stp`: source STEP design file.
- `basic_shapes/s_rice.xyz`: dense point and normal samples extracted from the STEP file.
- `camera_images/`: calibrated top/bottom camera images.

## Current Baseline

Run the initial XYZ view matching baseline:

```powershell
python scripts\match_xyz_views.py
```

The script renders candidate point-cloud views and writes match scores and overlay previews under `outputs/`.

## Design

See `docs/alignment_system_design.md` for the planned primitive extraction, geometry alignment, color alignment, and pose optimization pipeline.
