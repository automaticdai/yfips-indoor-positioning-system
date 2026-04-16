"""Tests for the rolling fps meter."""

from yfips.detection import FpsMeter


def test_first_tick_returns_zero():
    m = FpsMeter()
    assert m.tick(0.0) == 0.0


def test_steady_10hz():
    m = FpsMeter(window=10)
    rate = 0.0
    for i in range(10):
        rate = m.tick(i * 0.1)
    assert abs(rate - 10.0) < 0.05


def test_steady_60hz():
    m = FpsMeter(window=30)
    rate = 0.0
    for i in range(30):
        rate = m.tick(i * (1.0 / 60))
    assert abs(rate - 60.0) < 0.5


def test_window_drops_old_samples():
    # With window=3, only the most recent 3 timestamps count.
    m = FpsMeter(window=3)
    m.tick(0.0)
    m.tick(0.01)
    m.tick(0.02)
    m.tick(1.0)
    rate = m.tick(2.0)
    # Deque holds [0.02, 1.0, 2.0]; span=1.98, samples=2 → ~1.01 fps
    assert 0.9 < rate < 1.2


def test_zero_span_returns_zero():
    m = FpsMeter()
    m.tick(5.0)
    # Same timestamp → span is 0
    assert m.tick(5.0) == 0.0
