"""End-to-end tests for ImageRefDetector against synthetic ORB-textured scenes."""

import cv2
import numpy as np

from yfips.image_detector import ImageRefDetector


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


def test_detector_emits_expected_dict_shape(tmp_path):
    ref = _textured(seed=4)
    cv2.imwrite(str(tmp_path / "5.png"), ref)
    detector = ImageRefDetector(str(tmp_path), min_inliers=5)

    detections = detector.detect(_scene_with_ref(ref))

    assert detections, "ref should match in scene"
    det = detections[0]
    assert set(det.keys()) == {"id", "center", "forward", "corners"}
    assert det["corners"].shape == (4, 2)
    assert len(det["center"]) == 2
    assert len(det["forward"]) == 2
