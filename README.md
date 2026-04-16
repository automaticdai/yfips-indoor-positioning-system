# YF-IPS: Robot Indoor Positioning System
![](https://img.shields.io/github/stars/yfrobotics/yfips-indoor-positioning-system) ![](https://img.shields.io/github/issues/yfrobotics/yfips-indoor-positioning-system) ![](https://img.shields.io/github/license/yfrobotics/yfips-indoor-positioning-system)

## 1. Introduction
Traditional localisation systems are unfriendly to robot researchers and learners: they're expensive and take a lot of effort to set up. YF-IPS targets a low-cost indoor positioning system (IPS) with reasonable accuracy, usable for tracking and navigation beyond robotics as well.

The system is designed to locate 1–100 robots at ≥10 fps over a 5×5 m area, returning `(id, x, y, yaw)` — where `id` is the robot identifier, `(x, y)` is the 2D position relative to a world origin (in metres), and `yaw` is the rotation about z (in radians).

Two planned variants exist: vision-based and ToF/tag-based, usable together or separately. This repo currently implements the **vision** variant, with two selectable detection modes:

1. **AprilTag mode** — robots carry an AprilTag fiducial; detection uses the [swatbotics AprilTag](https://github.com/swatbotics/apriltag) binding.
2. **Reference-image mode** — no fiducials; each robot is localized by matching an overhead reference image against the live scene using ORB features + RANSAC homography.

## 2. Design specification
- Detect robots in a 5×5 m region.
- Simplest setup: ±20 cm precision. More sensors can improve to ±10 cm.
- 1–10 robots simultaneously at ≥10 fps.
- Output publishable as UDP JSON (implemented) or a ROS topic (planned).

## 3. Hardware
- **PC** (desktop or laptop) to run the program.
- **Webcam** — an HD (1080p) camera is recommended. A Logitech C920/C922 gives a good quality/cost tradeoff.
- **AprilTag printouts** — for AprilTag mode.
- **Reference images** — for image mode (top-down crops of each robot, one file per id).
- Wireless anchors/tags — TBD (ToF variant not yet implemented).

## 4. Install and run

Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync                                            # create .venv and install deps
uv run python -m yfips.calibration                 # one-time camera calibration
uv run python -m yfips.detection --mode apriltag   # or --mode image
```

Mode can also be set permanently via `"mode": "apriltag" | "image"` in `config.json`.

### UDP output
Detections are published as JSON datagrams `{"id", "x", "y", "yaw", "t"}` to `127.0.0.1:9999` by default (configurable under `udp` in `config.json`). Quick listener:

```bash
python3 -c "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind(('127.0.0.1',9999))
while True: print(s.recvfrom(4096)[0].decode())"
```

## 5. Calibration

### 5.1 Camera calibration
Capture ~15 images of a 7×6-inner-corner chessboard and save them as `images/calibration_*.jpg`. Run `uv run python -m yfips.calibration` — it writes the camera matrix and distortion coefficients into `config.json`.

### 5.2 Environment calibration
Run `uv run python -m yfips.detection` and **double-click the 4 world corners** of your playing area inside the video window, in the same order as `world_corners_m` in `config.json` (default: `(0,0), (5,0), (5,5), (0,5)` metres). The image→world homography is computed and persisted; subsequent runs reuse it. A 5th click resets and re-captures.

## 6. Detection modes

### 6.1 AprilTag mode (default)
Print AprilTags (any supported family), mount one per robot with the tag id corresponding to the robot id. Position is recovered via the world-plane homography (no PnP), so the printed side length doesn't affect localization — but keep tags large enough to detect reliably (≥5 cm for a 1080p camera at ~3 m height).

Tunables under `apriltag_mode` in `config.json` map onto `apriltag.DetectorOptions`:

```json
"apriltag_mode": {
  "family": "tag36h11",
  "nthreads": 4,
  "quad_decimate": 1.0,
  "refine_edges": true
}
```

`quad_decimate > 1.0` trades detection range for speed (downsamples before quad detection). `nthreads` parallelizes across cores.

### 6.2 Reference-image mode
Create a `references/` directory at the repo root and drop one image per robot:

```
references/
  1.png     # top-down image of robot with id=1
  2.png
  7.jpg
```

The filename stem must be the integer robot id. Requirements for good detection:
- Planar, textured top-down view of the robot (the feature matcher assumes a plane).
- Enough texture — flat colours or reflective surfaces produce few ORB features.
- Ideally the same approximate scale as the robot will appear in the scene.

Tunables under `image_mode` in `config.json`:
- `min_inliers` — RANSAC inlier threshold (default 15). Raise for robustness, lower for faint matches.
- `use_flann` — use FLANN-LSH instead of brute-force matcher (default `false`). Faster with many references.

## 7. Live visualizer

A second terminal can subscribe to the UDP stream and show each robot as an arrow on the world plane:

```bash
uv run python -m yfips.gui
```

## 8. Extras in `config.json`

- `undistort` — apply lens undistortion per frame using the calibrated intrinsics (default `true`).
- `tracker` — per-id smoothing across frames. Two types:
  - **EMA** (default, low overhead):
    ```json
    "tracker": { "enabled": true, "type": "ema", "alpha": 0.4, "timeout_s": 1.0 }
    ```
  - **Kalman** (constant-velocity model on `[x, y, yaw, vx, vy, ω]`; predicts through brief dropouts, wraps yaw residuals):
    ```json
    "tracker": {
      "enabled": true,
      "type": "kalman",
      "q_accel": 1.0,
      "r_pos": 0.05,
      "r_yaw": 0.1,
      "timeout_s": 1.0
    }
    ```
    Raise `q_accel` to react faster; raise `r_pos`/`r_yaw` to smooth harder. The filter re-initializes for an id if no measurement arrives within `timeout_s`.

    On frames where a tracked id is missing (briefly occluded, missed match), the Kalman tracker emits a UDP/ROS payload with `"predicted": true` derived from the constant-velocity model. EMA cannot extrapolate and stays silent on dropouts.
- `ros` — optional ROS 2 publisher: `{ "enabled": false, "topic": "/yfips/detections" }`. Requires `rclpy`; otherwise gracefully disables itself.

## 9. Credits
- Built on [OpenCV](https://opencv.org/).
- AprilTag detection via the swatbotics binding: https://github.com/swatbotics/apriltag

## 10. Contributors
- [automaticdai](https://github.com/automaticdai)
- [xinyu-xu-dev](https://github.com/xinyu-xu-dev)

## License
MIT
