from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CameraImageRecord:
    image_id: str
    path: Path
    group: str
    width: int
    height: int
    intrinsics: Optional[dict] = None
    distortion: Optional[dict] = None
    is_distortion_corrected: bool = True


@dataclass
class WeakPerspectivePose:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    scale_px_per_unit: float = 1.0
    tx_px: float = 0.0
    ty_px: float = 0.0
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model": "weak_perspective",
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "scale_px_per_unit": self.scale_px_per_unit,
            "tx_px": self.tx_px,
            "ty_px": self.ty_px,
            "score": self.score,
        }
