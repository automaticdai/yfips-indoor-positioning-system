import glob
import os

import cv2 as cv
import numpy as np

from yfips import config

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

objpoints = []
imgpoints = []

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
images = glob.glob(os.path.join(repo_root, "images", "calibration_*.jpg"))

if not images:
    raise SystemExit("No calibration images found at images/calibration_*.jpg")

gray = None
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        cv.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv.imshow("img", img)
        cv.waitKey(500)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("RMS reprojection error:", ret)
print("Camera matrix:\n", mtx)
print("Distortion coeffs:", dist.ravel())

cfg = config.load()
cfg["camera_matrix"] = mtx.tolist()
cfg["dist_coeffs"] = dist.ravel().tolist()
config.save(cfg)
print(f"Saved intrinsics to {config.CONFIG_PATH}")

cv.destroyAllWindows()
