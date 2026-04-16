from __future__ import annotations

import json
import os
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(_REPO_ROOT, "config.json")

DEFAULTS: dict[str, Any] = {
    "camera_matrix": None,
    "dist_coeffs": None,
    "world_corners_m": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]],
    "image_corners_px": None,
    "camera": {"index": 0, "width": 640, "height": 480, "fps": 60},
    "udp": {"host": "127.0.0.1", "port": 9999, "enabled": True},
    "mode": "apriltag",
    "references_dir": None,
    "image_mode": {"min_inliers": 15, "use_flann": False},
    "undistort": True,
    "tracker": {"enabled": True, "type": "ema", "alpha": 0.4, "timeout_s": 1.0,
                "q_accel": 1.0, "r_pos": 0.05, "r_yaw": 0.1},
    "ros": {"enabled": False, "topic": "/yfips/detections"},
}


def load() -> dict[str, Any]:
    if not os.path.exists(CONFIG_PATH):
        return dict(DEFAULTS)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    merged = dict(DEFAULTS)
    merged.update(cfg)
    return merged


def save(cfg: dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def update(**kwargs: Any) -> dict[str, Any]:
    cfg = load()
    cfg.update(kwargs)
    save(cfg)
    return cfg
