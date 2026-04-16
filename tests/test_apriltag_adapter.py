"""Tests for AprilTagAdapter's per-detection adaptation logic."""

import numpy as np

from yfips.detection import AprilTagAdapter


class _FakeDet:
    def __init__(self, tag_id, corners, center):
        self.tag_id = tag_id
        self.corners = corners
        self.center = center


def test_adapt_yields_forward_midpoint_of_corners_1_2():
    # axis-aligned tag: c0 (back-left), c1 (back-right), c2 (front-right), c3 (front-left)
    # forward should be midpoint of c1+c2 (right edge) per swatbotics convention
    corners = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    det = _FakeDet(tag_id=7, corners=corners, center=(5.0, 5.0))

    out = AprilTagAdapter._adapt(det)

    assert out["id"] == 7
    assert out["center"] == (5.0, 5.0)
    assert out["forward"] == (10.0, 5.0)
    np.testing.assert_array_equal(out["corners"], corners)


def test_adapt_id_is_python_int():
    corners = np.zeros((4, 2))
    det = _FakeDet(tag_id=np.int64(3), corners=corners, center=(0.0, 0.0))

    out = AprilTagAdapter._adapt(det)

    assert isinstance(out["id"], int)
    assert not isinstance(out["id"], np.integer)
