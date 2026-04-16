"""End-to-end tests for ImageRefDetector against synthetic ORB-textured scenes."""

import math

import cv2
import numpy as np

from yfips.image_detector import (
    ImageRefDetector,
    _dedup_by_id,
    _load_ref_meta,
    forward_point_in_ref,
)


def _textured(seed, size=(160, 160)):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=size, dtype=np.uint8)


def _scene_with_ref(ref, scene_size=(400, 400), at=(80, 80)):
    scene = np.full(scene_size, 128, dtype=np.uint8)
    h, w = ref.shape
    y0, x0 = at
    scene[y0:y0 + h, x0:x0 + w] = ref
    return scene


def test_detector_finds_planted_ref(tmp_path):
    ref = _textured(seed=42)
    cv2.imwrite(str(tmp_path / "7.png"), ref)
    detector = ImageRefDetector(str(tmp_path), min_inliers=5)

    detections = detector.detect(_scene_with_ref(ref))

    ids = {d["id"] for d in detections}
    assert 7 in ids


def test_detector_picks_correct_ref_among_two(tmp_path):
    ref_a = _textured(seed=1)
    ref_b = _textured(seed=2)
    cv2.imwrite(str(tmp_path / "1.png"), ref_a)
    cv2.imwrite(str(tmp_path / "2.png"), ref_b)
    detector = ImageRefDetector(str(tmp_path), min_inliers=5)

    detections = detector.detect(_scene_with_ref(ref_a))

    ids = {d["id"] for d in detections}
    assert 1 in ids
    assert 2 not in ids


def test_detector_returns_empty_on_blank_scene(tmp_path):
    cv2.imwrite(str(tmp_path / "1.png"), _textured(seed=3))
    detector = ImageRefDetector(str(tmp_path), min_inliers=5)

    blank = np.full((400, 400), 128, dtype=np.uint8)
    assert detector.detect(blank) == []


def test_dedup_by_id_keeps_highest_inliers():
    a = {"id": 7, "inliers": 20, "center": (0, 0)}
    b = {"id": 7, "inliers": 30, "center": (5, 5)}
    c = {"id": 8, "inliers": 25, "center": (1, 1)}
    out = _dedup_by_id([a, b, c])
    by_id = {d["id"]: d for d in out}
    assert set(by_id) == {7, 8}
    assert by_id[7]["inliers"] == 30
    assert by_id[7]["center"] == (5, 5)


def test_dedup_by_id_preserves_singletons():
    inputs = [{"id": 1, "inliers": 5}, {"id": 2, "inliers": 7}]
    out = _dedup_by_id(inputs)
    assert {d["id"] for d in out} == {1, 2}


def test_dedup_by_id_empty():
    assert _dedup_by_id([]) == []


def test_forward_point_default_is_right_edge():
    fx, fy = forward_point_in_ref(w=100, h=80, forward_deg=0)
    assert math.isclose(fx, 100)
    assert math.isclose(fy, 40)


def test_forward_point_at_90_points_up_in_image():
    # Image y grows down; "up" should give a y smaller than center.
    fx, fy = forward_point_in_ref(w=100, h=80, forward_deg=90)
    assert math.isclose(fx, 50)
    assert math.isclose(fy, -10)


def test_forward_point_at_180_is_left():
    fx, fy = forward_point_in_ref(w=100, h=80, forward_deg=180)
    assert math.isclose(fx, 0)
    assert math.isclose(fy, 40)


def test_forward_point_at_270_is_below():
    fx, fy = forward_point_in_ref(w=100, h=80, forward_deg=270)
    assert math.isclose(fx, 50)
    assert math.isclose(fy, 90)


def test_load_ref_meta_returns_default_when_no_sidecar(tmp_path):
    img = tmp_path / "7.png"
    img.write_bytes(b"")
    assert _load_ref_meta(str(img)) == {"forward_deg": 0.0}


def test_load_ref_meta_reads_sidecar(tmp_path):
    img = tmp_path / "7.png"
    img.write_bytes(b"")
    (tmp_path / "7.json").write_text('{"forward_deg": 90}')
    assert _load_ref_meta(str(img)) == {"forward_deg": 90.0}


def test_load_ref_meta_ignores_unknown_keys(tmp_path):
    img = tmp_path / "7.png"
    img.write_bytes(b"")
    (tmp_path / "7.json").write_text('{"forward_deg": 45, "foo": "bar"}')
    assert _load_ref_meta(str(img)) == {"forward_deg": 45.0}


def test_detector_emits_expected_dict_shape(tmp_path):
    ref = _textured(seed=4)
    cv2.imwrite(str(tmp_path / "5.png"), ref)
    detector = ImageRefDetector(str(tmp_path), min_inliers=5)

    detections = detector.detect(_scene_with_ref(ref))

    assert detections, "ref should match in scene"
    det = detections[0]
    assert set(det.keys()) == {"id", "center", "forward", "corners", "inliers"}
    assert det["corners"].shape == (4, 2)
    assert len(det["center"]) == 2
    assert len(det["forward"]) == 2
    assert det["inliers"] >= 5
