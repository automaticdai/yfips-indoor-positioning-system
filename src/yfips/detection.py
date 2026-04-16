import argparse
import json
import math
import os
import socket
import time

import cv2
import numpy as np

from yfips import config
from yfips.image_detector import ImageRefDetector
from yfips.kalman_tracker import KalmanTracker
from yfips.ros_publisher import RosPublisher
from yfips.tracker import EMATracker

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
WINDOW_NAME = "YFIPS"


class Publisher:
    _FINITE_FIELDS = ("x", "y", "yaw")

    def __init__(self, udp_cfg):
        self.enabled = udp_cfg.get("enabled", False)
        self.addr = (udp_cfg.get("host", "127.0.0.1"), int(udp_cfg.get("port", 9999)))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self.enabled else None
        self._warned_ids = set()

    def send(self, payload):
        if not self.enabled:
            return
        for field in self._FINITE_FIELDS:
            v = payload.get(field)
            if v is None or not math.isfinite(v):
                rid = payload.get("id")
                if rid not in self._warned_ids:
                    print(f"[publisher] dropping non-finite {field}={v!r} for id={rid}")
                    self._warned_ids.add(rid)
                return
        # allow_nan=False is a defensive backstop in case a non-numeric sneaks in.
        self.sock.sendto(json.dumps(payload, allow_nan=False).encode("utf-8"), self.addr)


def compute_homography(image_corners_px, world_corners_m):
    src = np.array(image_corners_px, dtype=np.float32)
    dst = np.array(world_corners_m, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def image_to_world(H, pt):
    p = np.array([pt[0], pt[1], 1.0])
    w = H @ p
    return float(w[0] / w[2]), float(w[1] / w[2])


class CalibClicker:
    def __init__(self, cfg):
        self.cfg = cfg
        existing = cfg.get("image_corners_px")
        self.points = list(existing) if existing else []
        self.H = None
        if len(self.points) == 4:
            self.H = compute_homography(self.points, cfg["world_corners_m"])

    def __call__(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDBLCLK:
            return
        if len(self.points) >= 4:
            self.points = []
            self.H = None
        self.points.append([x, y])
        print(f"calib point {len(self.points)}/4: ({x},{y})")
        if len(self.points) == 4:
            self.H = compute_homography(self.points, self.cfg["world_corners_m"])
            self.cfg["image_corners_px"] = self.points
            config.save(self.cfg)
            print("Homography computed and saved.")


class AprilTagAdapter:
    """Wraps the apriltag library into the common detection dict shape."""

    def __init__(self):
        import apriltag  # imported lazily so image-mode users don't need it
        self.detector = apriltag.Detector()

    @staticmethod
    def _adapt(det):
        # swatbotics corner order: 0 back-left, 1 back-right, 2 front-right, 3 front-left.
        # The right edge (midpoint of c1+c2) sits ahead of center along the tag's +x.
        c = np.asarray(det.corners)
        forward = 0.5 * (c[1] + c[2])
        return {
            "id": int(det.tag_id),
            "center": tuple(det.center),
            "forward": (float(forward[0]), float(forward[1])),
            "corners": c,
        }

    def detect(self, gray):
        return [self._adapt(det) for det in self.detector.detect(gray)]


def build_detector(mode, cfg):
    if mode == "apriltag":
        return AprilTagAdapter()
    if mode == "image":
        ref_dir = cfg.get("references_dir") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "references")
        im_cfg = cfg.get("image_mode", {})
        return ImageRefDetector(
            ref_dir,
            min_inliers=int(im_cfg.get("min_inliers", 15)),
            use_flann=bool(im_cfg.get("use_flann", False)),
        )
    raise ValueError(f"unknown mode: {mode}")


def yaw_from_forward(H, center_px, forward_px):
    cw = image_to_world(H, center_px)
    fw = image_to_world(H, forward_px)
    return math.atan2(fw[1] - cw[1], fw[0] - cw[0])


def emit_predictions(tracker, detected_ids, t):
    """Yield (rid, x, y, yaw) for tracked ids absent this frame whose
    tracker can extrapolate. Trackers without a velocity model (EMA)
    silently produce nothing."""
    if tracker is None:
        return
    for rid in tracker.ids():
        if rid in detected_ids:
            continue
        out = tracker.predict_only(rid, t)
        if out is None:
            continue
        yield (rid, *out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["apriltag", "image"], default=None,
                        help="detection mode; overrides config.mode")
    args = parser.parse_args()

    cfg = config.load()
    mode = args.mode or cfg.get("mode", "apriltag")
    print(f"[yfips] mode={mode}")

    pub = Publisher(cfg["udp"])
    ros_pub = RosPublisher(cfg.get("ros", {}))
    tracker_cfg = cfg.get("tracker", {})
    if not tracker_cfg.get("enabled", True):
        tracker = None
    elif tracker_cfg.get("type", "ema") == "kalman":
        tracker = KalmanTracker(
            q_accel=float(tracker_cfg.get("q_accel", 1.0)),
            r_pos=float(tracker_cfg.get("r_pos", 0.05)),
            r_yaw=float(tracker_cfg.get("r_yaw", 0.1)),
            timeout_s=float(tracker_cfg.get("timeout_s", 1.0)),
        )
    else:
        tracker = EMATracker(
            alpha=float(tracker_cfg.get("alpha", 0.4)),
            timeout_s=float(tracker_cfg.get("timeout_s", 1.0)),
        )
    clicker = CalibClicker(cfg)
    detector = build_detector(mode, cfg)

    undistort_maps = None
    if cfg.get("undistort", True) and cfg.get("camera_matrix") and cfg.get("dist_coeffs"):
        K = np.array(cfg["camera_matrix"], dtype=np.float32)
        D = np.array(cfg["dist_coeffs"], dtype=np.float32)
        size = (IMAGE_WIDTH, IMAGE_HEIGHT)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, size, alpha=0.0, newImgSize=size)
        undistort_maps = cv2.initUndistortRectifyMap(K, D, None, new_K, size, cv2.CV_16SC2)
        print("[yfips] live undistortion enabled")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, clicker)

    while True:
        now = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        if undistort_maps is not None:
            frame = cv2.remap(frame, undistort_maps[0], undistort_maps[1],
                              cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for p in clicker.points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, (255, 0, 0), -1)

        detections = detector.detect(gray)
        detected_ids = set()
        for det in detections:
            cx, cy = det["center"]
            for cn in det["corners"]:
                cv2.circle(frame, (int(cn[0]), int(cn[1])), 3, (0, 0, 255), -1)
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
            fx_, fy_ = det["forward"]
            cv2.arrowedLine(frame, (int(cx), int(cy)), (int(fx_), int(fy_)),
                            (0, 255, 255), 1, tipLength=0.3)

            if clicker.H is not None:
                x_w, y_w = image_to_world(clicker.H, det["center"])
                yaw = yaw_from_forward(clicker.H, det["center"], det["forward"])
                if tracker is not None:
                    x_w, y_w, yaw = tracker.update(det["id"], x_w, y_w, yaw, now)
                detected_ids.add(det["id"])
                label = f"id={det['id']} x={x_w:.2f} y={y_w:.2f} yaw={math.degrees(yaw):.0f}"
                cv2.putText(frame, label, (int(cx) + 6, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                payload = {"id": det["id"], "x": x_w, "y": y_w,
                           "yaw": yaw, "t": now}
                pub.send(payload)
                ros_pub.send(payload)
            else:
                cv2.putText(frame, f"id={det['id']}", (int(cx) + 6, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if clicker.H is not None:
            for rid, x_w, y_w, yaw in emit_predictions(tracker, detected_ids, now):
                payload = {"id": rid, "x": x_w, "y": y_w,
                           "yaw": yaw, "t": now, "predicted": True}
                pub.send(payload)
                ros_pub.send(payload)

        fps = 1.0 / max(time.time() - now, 1e-6)
        cv2.putText(frame, f"{mode} | fps: {fps:.1f}",
                    (0, IMAGE_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
        if clicker.H is None:
            cv2.putText(frame, "double-click 4 corners in world_corners_m order",
                        (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255))

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    ros_pub.shutdown()


if __name__ == "__main__":
    main()
