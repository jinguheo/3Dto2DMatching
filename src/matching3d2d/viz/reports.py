from __future__ import annotations

import json
from pathlib import Path


def save_pose_json(pose: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        k: round(float(v), 5) if isinstance(v, float) else v
        for k, v in pose.items()
    }
    if "model" not in record:
        record = {"model": "weak_perspective", **record}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
