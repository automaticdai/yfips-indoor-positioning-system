# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

YF-IPS: a low-cost indoor positioning system for tracking 1–100 robots in a ~5×5 m area at ≥10 fps, returning `(id, x, y, yaw)`. Vision-based (webcam) with two detection modes — **AprilTag** and **reference-image (ORB)**. Early-stage research code.

## Setup & Run

Package management uses [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` (`requirements.txt` is kept for reference but no longer authoritative).

```bash
uv sync                                           # create .venv + install deps
uv run python src/calibration.py                  # chessboard calibration → config.json
uv run python src/detection.py --mode apriltag    # default mode
uv run python src/detection.py --mode image       # reference-image mode
```

Mode can also be set via `"mode"` in `config.json`. Add deps with `uv add <pkg>`.

Inside the window: double-click 4 world corners in the order of `world_corners_m` (default `(0,0), (5,0), (5,5), (0,5)` m) to compute and persist the image→world homography. 5th click resets. ESC quits.

No tests or linter are configured.

## Architecture

All state flows through `config.json` (at repo root). Each script in `src/` is runnable standalone; there is no package-level entry point.

- `src/config.py` — load/save `config.json`. Holds camera intrinsics, distortion, world/image corners, tag size, UDP settings, mode, references dir.
- `src/calibration.py` — OpenCV chessboard calibration. Reads `images/calibration_*.jpg` (absolute-path glob; CWD-independent). Writes `camera_matrix` + `dist_coeffs` into `config.json`.
- `src/detection.py` — main realtime loop.
  - Builds a detector based on `mode` (`apriltag` uses the `apriltag` lib; `image` uses `ImageRefDetector`).
  - Mouse double-click collects 4 world-corner pixels → `cv2.findHomography` → image→world homography, persisted.
  - Per detection: center + a "forward" image point are mapped through the homography; yaw = `atan2` of the world-frame forward vector.
  - Publishes `{id, x, y, yaw, t}` JSON over UDP (default `127.0.0.1:9999`).
- `src/image_detector.py` — `ImageRefDetector`: ORB + BFMatcher + RANSAC homography. Loads references from `references/<id>.{png,jpg}` where the filename stem is the integer robot id. Emits the same `{id, center, forward, corners}` shape as the AprilTag adapter so downstream world-transform code is shared.
- `src/gui.py` — PySide2 "Hello World" stub; future Qt frontend.

UDP listener (debug):
```bash
python3 -c "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind(('127.0.0.1',9999))
while True: print(s.recvfrom(4096)[0].decode())"
```

## Gotchas

- `apriltag` package name on PyPI varies by platform (swatbotics binding).
- `pyside2` wheels don't exist for Python ≥3.11; if `uv sync` fails on PySide2, constrain `requires-python` in `pyproject.toml` or swap to `pyside6`.
- `detection.py` hardcodes camera index 0 and 640×480@60 fps.
- Image mode cost scales linearly with number of references; for ≳50 robots swap BFMatcher for FLANN.
- No live undistortion — intrinsics are saved but not applied before detection. The homography absorbs distortion near the clicked corners only.
- World positioning assumes robots move on a **plane**; tall tags/robots get parallax error.
