"""Tests for the camera-open error path."""

import pytest

from yfips import detection


class _FakeCap:
    def __init__(self, opened=True):
        self._opened = opened
        self.props = {}

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        self.props[prop] = value
        return True


def test_open_camera_raises_when_not_opened(monkeypatch):
    monkeypatch.setattr(detection.cv2, "VideoCapture", lambda idx: _FakeCap(opened=False))
    with pytest.raises(SystemExit, match="index 3"):
        detection.open_camera({"index": 3, "width": 640, "height": 480, "fps": 60})


def test_open_camera_returns_cap_when_opened(monkeypatch):
    fake = _FakeCap(opened=True)
    monkeypatch.setattr(detection.cv2, "VideoCapture", lambda idx: fake)
    cap = detection.open_camera({"index": 0, "width": 1280, "height": 720, "fps": 30})
    assert cap is fake
    assert fake.props[detection.cv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert fake.props[detection.cv2.CAP_PROP_FRAME_HEIGHT] == 720
    assert fake.props[detection.cv2.CAP_PROP_FPS] == 30
