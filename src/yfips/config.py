import json
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(_REPO_ROOT, "config.json")

DEFAULTS = {
    "camera_matrix": None,
    "dist_coeffs": None,
    "world_corners_m": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]],
    "image_corners_px": None,
    "tag_size_m": 0.10,
    "udp": {"host": "127.0.0.1", "port": 9999, "enabled": True},
    "mode": "apriltag",
    "references_dir": None,
    "image_mode": {"min_inliers": 15, "use_flann": False},
    "undistort": True,
    "tracker": {"enabled": True, "type": "ema", "alpha": 0.4, "timeout_s": 1.0,
                "q_accel": 1.0, "r_pos": 0.05, "r_yaw": 0.1},
    "ros": {"enabled": False, "topic": "/yfips/detections"},
}


def load():
    if not os.path.exists(CONFIG_PATH):
        return dict(DEFAULTS)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    merged = dict(DEFAULTS)
    merged.update(cfg)
    return merged


def save(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def update(**kwargs):
    cfg = load()
    cfg.update(kwargs)
    save(cfg)
    return cfg
