"""Per-id constant-velocity Kalman tracker over (x, y, yaw).

State:        [x, y, yaw, vx, vy, w]^T
Measurement:  [x, y, yaw]^T (yaw residual is wrapped to [-pi, pi])
Process model: constant velocity, Gaussian acceleration noise.
"""

import math

import numpy as np


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class _Filter:
    def __init__(self, x, y, yaw, t, q_accel, r_pos, r_yaw):
        self.x = np.array([x, y, yaw, 0.0, 0.0, 0.0])
        self.P = np.diag([0.1, 0.1, 0.1, 1.0, 1.0, 1.0])
        self.t = t
        self.q_accel = q_accel
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
        self.R = np.diag([r_pos ** 2, r_pos ** 2, r_yaw ** 2])

    def _F_Q(self, dt):
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        q = self.q_accel ** 2
        # discrete-time white-noise acceleration covariance, decoupled per axis
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        block = np.array([[dt4 / 4, dt3 / 2], [dt3 / 2, dt2]]) * q
        Q = np.zeros((6, 6))
        for i in (0, 1, 2):
            Q[np.ix_([i, i + 3], [i, i + 3])] = block
        return F, Q

    def predict(self, t):
        dt = max(t - self.t, 1e-3)
        F, Q = self._F_Q(dt)
        self.x = F @ self.x
        self.x[2] = _wrap(self.x[2])
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, zx, zy, zyaw):
        z = np.array([zx, zy, zyaw])
        y = z - self.H @ self.x
        y[2] = _wrap(y[2])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = _wrap(self.x[2])
        self.P = (np.eye(6) - K @ self.H) @ self.P


class KalmanTracker:
    def __init__(self, q_accel=1.0, r_pos=0.05, r_yaw=0.1, timeout_s=1.0):
        self.q_accel = q_accel
        self.r_pos = r_pos
        self.r_yaw = r_yaw
        self.timeout_s = timeout_s
        self.filters = {}  # id -> _Filter

    def update(self, rid, x, y, yaw, t):
        f = self.filters.get(rid)
        if f is None or (t - f.t) > self.timeout_s:
            f = _Filter(x, y, yaw, t, self.q_accel, self.r_pos, self.r_yaw)
            self.filters[rid] = f
        else:
            f.predict(t)
            f.update(x, y, yaw)
        return float(f.x[0]), float(f.x[1]), float(f.x[2])
