"""Reference-image detector: locate robots by matching scene features to
per-robot reference images. File stem must be the integer robot id
(e.g. references/7.png → id=7). Returns image-space center + a point one
unit along the reference's local +x axis so downstream code can reuse the
same world-homography path as the AprilTag mode."""

import glob
import os

import cv2
import numpy as np


class ImageRefDetector:
    def __init__(self, ref_dir, min_inliers=15, ratio=0.75,
                 max_features=2000, use_flann=False):
        self.min_inliers = min_inliers
        self.ratio = ratio
        self.orb = cv2.ORB_create(max_features)
        if use_flann:
            index_params = dict(algorithm=6, table_number=12,
                                key_size=20, multi_probe_level=2)
            self.matcher = cv2.FlannBasedMatcher(index_params, dict(checks=50))
            print("[image_detector] using FLANN-LSH matcher")
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.refs = []  # list of (id, keypoints, descriptors, (w, h))

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
            self.refs.append((rid, kp, des, (w, h)))
            print(f"[image_detector] loaded ref id={rid} ({w}x{h}, {len(kp)} kp)")

    def detect(self, gray):
        results = []
        if not self.refs:
            return results
        kp2, des2 = self.orb.detectAndCompute(gray, None)
        if des2 is None or len(kp2) < self.min_inliers:
            return results

        for rid, kp1, des1, (w, h) in self.refs:
            knn = self.matcher.knnMatch(des1, des2, k=2)
            good = [m for pair in knn if len(pair) == 2
                    for m, n in [pair] if m.distance < self.ratio * n.distance]
            if len(good) < self.min_inliers:
                continue
            src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None or int(mask.sum()) < self.min_inliers:
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
            })
        return results
