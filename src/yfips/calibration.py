"""Camera intrinsic calibration from a chessboard pattern.

Reads images/calibration_*.jpg at the repo root, runs OpenCV's chessboard
corner finder + calibrateCamera, and writes the intrinsics into
config.json.
"""

from __future__ import annotations

import glob
import os
from typing import Any

import cv2 as cv
import numpy as np

from yfips import config

CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
CHESSBOARD = (7, 6)  # inner corners (cols, rows)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_IMAGES_GLOB = os.path.join(_REPO_ROOT, "images", "calibration_*.jpg")


def _make_object_points() -> np.ndarray:
    cols, rows = CHESSBOARD
    pts = np.zeros((rows * cols, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return pts


def apply_intrinsics(cfg: dict[str, Any], mtx: Any, dist: Any) -> dict[str, Any]:
    """Update cfg in place with new intrinsics, dropping any image_corners_px
    that were collected under the previous calibration. The previously
    saved homography no longer matches the rectified image, so the user
    must re-click."""
    cfg["camera_matrix"] = mtx
    cfg["dist_coeffs"] = dist
    if cfg.get("image_corners_px") is not None:
        print("[calibration] WARNING: image_corners_px cleared — "
              "re-click 4 world corners after restarting detection")
        cfg["image_corners_px"] = None
    return cfg


def main() -> None:
    images = glob.glob(_IMAGES_GLOB)
    if not images:
        raise SystemExit(f"No calibration images found at {_IMAGES_GLOB}")

    objp = _make_object_points()
    objpoints, imgpoints = [], []
    gray = None
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            imgpoints.append(corners2)
            cv.drawChessboardCorners(img, CHESSBOARD, corners2, ret)
            cv.imshow("img", img)
            cv.waitKey(500)

    ret, mtx, dist, _, _ = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("RMS reprojection error:", ret)
    print("Camera matrix:\n", mtx)
    print("Distortion coeffs:", dist.ravel())

    cfg = config.load()
    apply_intrinsics(cfg, mtx=mtx.tolist(), dist=dist.ravel().tolist())
    config.save(cfg)
    print(f"Saved intrinsics to {config.CONFIG_PATH}")

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
