# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

YF-IPS: a low-cost indoor positioning system for tracking 1–100 robots in a ~5×5 m area at ≥10 fps, returning `(id, x, y, yaw)`. Vision-based (webcam) with two detection modes — **AprilTag** and **reference-image (ORB)**. Early-stage research code.

## Setup & Run

Package management uses [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` (`requirements.txt` is kept for reference but no longer authoritative).

```bash
uv sync                                               # create .venv + install deps
uv run python -m yfips.calibration                    # chessboard calibration → config.json
uv run python -m yfips.detection --mode apriltag      # default mode
uv run python -m yfips.detection --mode image         # reference-image mode
uv run pytest                                         # unit tests
uv run ruff check .                                   # lint
```

Mode can also be set via `"mode"` in `config.json`. Add deps with `uv add <pkg>` (runtime) or `uv add --group dev <pkg>` (dev).

Inside the window: double-click 4 world corners in the order of `world_corners_m` (default `(0,0), (5,0), (5,5), (0,5)` m) to compute and persist the image→world homography. 5th click resets. ESC quits.

## Architecture

All state flows through `config.json` (at repo root). Source lives under `src/yfips/` as an installable package — run modules with `python -m yfips.<name>`.

- `src/yfips/config.py` — load/save `config.json`. Holds camera intrinsics, distortion, world/image corners, UDP settings, mode, references dir.
- `src/yfips/calibration.py` — OpenCV chessboard calibration. Reads `images/calibration_*.jpg` (absolute-path glob; CWD-independent). Writes `camera_matrix` + `dist_coeffs` into `config.json`.
- `src/yfips/detection.py` — main realtime loop.
  - Builds a detector based on `mode` (`apriltag` uses the `apriltag` lib; `image` uses `ImageRefDetector`).
  - If intrinsics are present and `undistort=true`, precomputes rectify maps (`cv2.initUndistortRectifyMap`) and remaps every frame.
  - Mouse double-click collects 4 world-corner pixels → `cv2.findHomography` → image→world homography, persisted.
  - Per detection: center + a "forward" image point are mapped through the homography; yaw = `atan2` of the world-frame forward vector. Smoothed by `EMATracker` per id before publish.
  - Publishes `{id, x, y, yaw, t}` JSON over UDP (default `127.0.0.1:9999`) and optionally as a ROS 2 string topic.
- `src/yfips/image_detector.py` — `ImageRefDetector`: ORB + BFMatcher/FLANN + RANSAC homography. Loads references from `references/<id>.{png,jpg}` where the filename stem is the integer robot id. Emits the same `{id, center, forward, corners}` shape as the AprilTag adapter so downstream world-transform code is shared. Set `image_mode.use_flann=true` for LSH-based matching at scale.
- `src/yfips/tracker.py` — `EMATracker`: per-id exponential-moving-average smoothing of `(x, y, yaw)`; yaw smoothed via unit vector to handle wrap-around. Disable via `tracker.enabled=false`.
- `src/yfips/ros_publisher.py` — optional ROS 2 publisher (std_msgs/String JSON). No-ops if `rclpy` isn't installed. Enable via `ros.enabled=true`.
- `src/yfips/gui.py` — matplotlib live visualizer; listens to the UDP stream and plots each tracked robot as an arrow on the world plane. Run in a second terminal: `uv run python -m yfips.gui`.
- `tests/` — pytest unit tests over pure-logic modules (trackers, homography, guards).

UDP listener (debug):
```bash
python3 -c "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind(('127.0.0.1',9999))
while True: print(s.recvfrom(4096)[0].decode())"
```

## Gotchas

- `apriltag` package name on PyPI varies by platform (swatbotics binding).
- Running `yfips.calibration` auto-clears any previously-saved `image_corners_px`, because the old pixel coordinates refer to an image rectified with the old intrinsics. Re-click the 4 world corners after calibration.
- Camera index, resolution and fps default to 0 / 640×480 / 60 fps but live in `config.json`'s `camera` block; CLI flags `--camera-index --width --height --fps` override at runtime.
- Image mode cost scales linearly with number of references; for ≳50 robots swap BFMatcher for FLANN.
- World positioning assumes robots move on a **plane**; tall tags/robots get parallax error even after undistortion.
- `ros` mode requires a ROS 2 install (Humble+) with `rclpy` on PYTHONPATH; otherwise it no-ops with a warning.
- Toggling `undistort` after clicking world corners still misaligns them — only a recalibration auto-invalidates them. If you flip `undistort` manually, re-click.
