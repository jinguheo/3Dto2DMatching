from __future__ import annotations

import math

import numpy as np


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
