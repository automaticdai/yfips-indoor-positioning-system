import argparse
import json
import math
import os
import socket
import time

import cv2
import numpy as np

import config
from image_detector import ImageRefDetector

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
WINDOW_NAME = "YFIPS"


class Publisher:
    def __init__(self, udp_cfg):
        self.enabled = udp_cfg.get("enabled", False)
        self.addr = (udp_cfg.get("host", "127.0.0.1"), int(udp_cfg.get("port", 9999)))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self.enabled else None

    def send(self, payload):
        if not self.enabled:
            return
        self.sock.sendto(json.dumps(payload).encode("utf-8"), self.addr)


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

    def detect(self, gray):
        out = []
        for det in self.detector.detect(gray):
            c = np.asarray(det.corners)
            # corner0→corner1 defines the tag's local +x
            forward = 0.5 * (c[1] + c[2])
            back = 0.5 * (c[0] + c[3])
            # forward_px should be ahead of center along +x
            out.append({
                "id": int(det.tag_id),
                "center": tuple(det.center),
                "forward": tuple(forward + (forward - back) * 0.0),  # == forward
                "corners": c,
            })
        return out


def build_detector(mode, cfg):
    if mode == "apriltag":
        return AprilTagAdapter()
    if mode == "image":
        ref_dir = cfg.get("references_dir") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "references")
        return ImageRefDetector(
            ref_dir,
            min_inliers=int(cfg.get("image_mode", {}).get("min_inliers", 15)),
        )
    raise ValueError(f"unknown mode: {mode}")


def yaw_from_forward(H, center_px, forward_px):
    cw = image_to_world(H, center_px)
    fw = image_to_world(H, forward_px)
    return math.atan2(fw[1] - cw[1], fw[0] - cw[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["apriltag", "image"], default=None,
                        help="detection mode; overrides config.mode")
    args = parser.parse_args()

    cfg = config.load()
    mode = args.mode or cfg.get("mode", "apriltag")
    print(f"[yfips] mode={mode}")

    pub = Publisher(cfg["udp"])
    clicker = CalibClicker(cfg)
    detector = build_detector(mode, cfg)

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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for p in clicker.points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, (255, 0, 0), -1)

        detections = detector.detect(gray)
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
                label = f"id={det['id']} x={x_w:.2f} y={y_w:.2f} yaw={math.degrees(yaw):.0f}"
                cv2.putText(frame, label, (int(cx) + 6, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                pub.send({"id": det["id"], "x": x_w, "y": y_w,
                          "yaw": yaw, "t": now})
            else:
                cv2.putText(frame, f"id={det['id']}", (int(cx) + 6, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

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


if __name__ == "__main__":
    main()
