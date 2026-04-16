"""Precedence tests for resolving camera settings (CLI > config > default)."""

from types import SimpleNamespace

from yfips.detection import camera_settings


def _args(**overrides):
    base = {"camera_index": None, "width": None, "height": None, "fps": None}
    base.update(overrides)
    return SimpleNamespace(**base)


def test_uses_defaults_when_nothing_set():
    s = camera_settings({}, _args())
    assert s == {"index": 0, "width": 640, "height": 480, "fps": 60}


def test_uses_config_when_cli_unset():
    cfg = {"camera": {"index": 2, "width": 1280, "height": 720, "fps": 30}}
    s = camera_settings(cfg, _args())
    assert s == {"index": 2, "width": 1280, "height": 720, "fps": 30}


def test_cli_overrides_config():
    cfg = {"camera": {"index": 0, "width": 640, "height": 480, "fps": 30}}
    s = camera_settings(cfg, _args(camera_index=1, width=800, height=600, fps=60))
    assert s == {"index": 1, "width": 800, "height": 600, "fps": 60}


def test_partial_cli_override():
    cfg = {"camera": {"index": 0, "width": 640, "height": 480, "fps": 30}}
    s = camera_settings(cfg, _args(width=1920, height=1080))
    assert s == {"index": 0, "width": 1920, "height": 1080, "fps": 30}
