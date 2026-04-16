"""Reference-image detector: locate robots by matching scene features to
per-robot reference images. File stem must be the integer robot id
(e.g. references/7.png → id=7). Returns image-space center + a point one
unit along the reference's local +x axis so downstream code can reuse the
same world-homography path as the AprilTag mode.

Each reference owns its own matcher, trained once at __init__. Querying
the scene against pre-trained matchers avoids rebuilding a FLANN-LSH
index every frame, which was the dominant cost when scaling past a few
references."""

import glob
import os

import cv2
import numpy as np


def _dedup_by_id(results):
    """Collapse duplicate-id detections to the one with the most inliers.
    Order is not preserved."""
    best = {}
    for det in results:
        prev = best.get(det["id"])
        if prev is None or det.get("inliers", 0) > prev.get("inliers", 0):
            best[det["id"]] = det
    return list(best.values())


class ImageRefDetector:
    def __init__(self, ref_dir, min_inliers=15, ratio=0.75,
                 max_features=2000, use_flann=False):
        self.min_inliers = min_inliers
        self.ratio = ratio
        self.use_flann = use_flann
        self.orb = cv2.ORB_create(max_features)
        self.refs = []  # list of (id, keypoints, matcher, (w, h))

        if use_flann:
            print("[image_detector] using FLANN-LSH matcher")

        if not os.path.isdir(ref_dir):
            print(f"[image_detector] reference dir missing: {ref_dir}")
            return

        for path in sorted(glob.glob(os.path.join(ref_dir, "*"))):
            stem = os.path.splitext(os.path.basename(path))[0]
            try:
                rid = int(stem)
            except ValueError:
                continue
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            kp, des = self.orb.detectAndCompute(img, None)
            if des is None or len(kp) < self.min_inliers:
                continue
            h, w = img.shape
            matcher = self._build_matcher()
            matcher.add([des])
            matcher.train()
            self.refs.append((rid, kp, matcher, (w, h)))
            print(f"[image_detector] loaded ref id={rid} ({w}x{h}, {len(kp)} kp)")

    def _build_matcher(self):
        if self.use_flann:
            index_params = dict(algorithm=6, table_number=12,
                                key_size=20, multi_probe_level=2)
            return cv2.FlannBasedMatcher(index_params, dict(checks=50))
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect(self, gray):
        results = []
        if not self.refs:
            return results
        kp2, des2 = self.orb.detectAndCompute(gray, None)
        if des2 is None or len(kp2) < self.min_inliers:
            return results

        for rid, kp1, matcher, (w, h) in self.refs:
            # Scene is the query side; ref descriptors are pre-trained.
            # m.queryIdx -> scene kp, m.trainIdx -> ref kp.
            knn = matcher.knnMatch(des2, k=2)
            good = [m for pair in knn if len(pair) == 2
                    for m, n in [pair] if m.distance < self.ratio * n.distance]
            if len(good) < self.min_inliers:
                continue
            src = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            inliers = int(mask.sum()) if mask is not None else 0
            if H is None or inliers < self.min_inliers:
                continue

            pts_ref = np.array([[[w / 2, h / 2]], [[w, h / 2]]], dtype=np.float32)
            pts_scene = cv2.perspectiveTransform(pts_ref, H)
            center = tuple(pts_scene[0, 0])
            forward = tuple(pts_scene[1, 0])

            corners_ref = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]],
                                   dtype=np.float32)
            corners_scene = cv2.perspectiveTransform(corners_ref, H).reshape(-1, 2)

            results.append({
                "id": rid,
                "center": center,
                "forward": forward,
                "corners": corners_scene,
                "inliers": inliers,
            })
        return _dedup_by_id(results)
