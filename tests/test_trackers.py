"""Tests for tracker prediction-only API and id enumeration."""


from yfips.detection import emit_predictions
from yfips.kalman_tracker import KalmanTracker
from yfips.tracker import EMATracker


def test_ema_ids_lists_tracked():
    ema = EMATracker()
    ema.update(1, 0.0, 0.0, 0.0, t=0.0)
    ema.update(7, 0.0, 0.0, 0.0, t=0.0)
    assert set(ema.ids()) == {1, 7}


def test_ema_predict_only_returns_none():
    ema = EMATracker()
    ema.update(1, 0.0, 0.0, 0.0, t=0.0)
    # EMA can't extrapolate without a velocity model
    assert ema.predict_only(1, t=1.0) is None


def test_kalman_ids_lists_tracked():
    k = KalmanTracker()
    k.update(1, 0.0, 0.0, 0.0, t=0.0)
    assert k.ids() == [1]


def test_kalman_predict_only_unknown_id_returns_none():
    k = KalmanTracker()
    assert k.predict_only(99, t=0.0) is None


def test_kalman_predict_only_after_timeout_returns_none():
    k = KalmanTracker(timeout_s=0.5)
    k.update(7, 0.0, 0.0, 0.0, t=0.0)
    # 1.0s later, exceeds timeout
    assert k.predict_only(7, t=1.0) is None


def test_kalman_predict_only_extrapolates_constant_velocity():
    # Drive the filter with two measurements spaced 1s apart with vx=1.
    # Then ask for a prediction 0.5s into the future and check x advanced.
    k = KalmanTracker(q_accel=0.05, r_pos=0.01, r_yaw=0.01, timeout_s=10.0)
    k.update(7, 0.0, 0.0, 0.0, t=0.0)
    x_at_meas, _, _ = k.update(7, 1.0, 0.0, 0.0, t=1.0)
    x_pred, y_pred, yaw_pred = k.predict_only(7, t=1.5)
    # Predict-only should move x forward; tolerance is loose since the
    # filter hasn't fully converged on velocity after only two updates.
    assert x_pred > x_at_meas
    assert abs(y_pred) < 0.1
    assert abs(yaw_pred) < 0.1


def test_emit_predictions_skips_detected_ids():
    k = KalmanTracker(timeout_s=10.0)
    k.update(1, 0.0, 0.0, 0.0, t=0.0)
    k.update(2, 0.0, 0.0, 0.0, t=0.0)
    out = list(emit_predictions(k, detected_ids={1}, t=0.5))
    assert [row[0] for row in out] == [2]


def test_emit_predictions_with_none_tracker_yields_nothing():
    assert list(emit_predictions(None, detected_ids=set(), t=0.0)) == []


def test_emit_predictions_with_ema_yields_nothing():
    ema = EMATracker()
    ema.update(1, 0.0, 0.0, 0.0, t=0.0)
    # EMA can't extrapolate, so even a missing id should produce no output.
    assert list(emit_predictions(ema, detected_ids=set(), t=0.5)) == []
