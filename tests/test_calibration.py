"""Tests for the pure-logic helpers in yfips.calibration."""

from yfips.calibration import apply_intrinsics


def test_apply_intrinsics_writes_matrix_and_dist():
    cfg = {"camera_matrix": None, "dist_coeffs": None}
    out = apply_intrinsics(cfg, mtx=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                           dist=[0.1, 0.0, 0.0, 0.0, 0.0])
    assert out["camera_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert out["dist_coeffs"] == [0.1, 0.0, 0.0, 0.0, 0.0]


def test_apply_intrinsics_clears_stale_image_corners():
    # Existing image_corners_px were clicked under the previous
    # intrinsics; after recalibration they no longer match the
    # rectified image. Drop them so the user re-clicks.
    cfg = {
        "camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "image_corners_px": [[10, 20], [30, 40], [50, 60], [70, 80]],
    }
    out = apply_intrinsics(cfg, mtx=[[2, 0, 0], [0, 2, 0], [0, 0, 1]],
                           dist=[0.5, 0.0, 0.0, 0.0, 0.0])
    assert out["image_corners_px"] is None


def test_apply_intrinsics_preserves_other_keys():
    cfg = {"world_corners_m": [[0, 0], [5, 5]], "udp": {"port": 9999}}
    out = apply_intrinsics(cfg, mtx=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                           dist=[0.0])
    assert out["world_corners_m"] == [[0, 0], [5, 5]]
    assert out["udp"] == {"port": 9999}


def test_apply_intrinsics_returns_same_dict_object():
    # In-place update is fine — calibration.main() owns the dict.
    cfg = {}
    out = apply_intrinsics(cfg, mtx=[[1, 0, 0]], dist=[0.0])
    assert out is cfg
