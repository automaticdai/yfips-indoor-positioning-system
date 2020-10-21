import apriltag
import numpy as np
import cv2

# this will be started as a background thread
# fiducial / AprilTag / ArUco

# campture from the camera
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # convert into gray scale image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('VisionNav - Robot Positioning and Navigation', img)

    detector = apriltag.Detector()
    result = detector.detect(img)

    print(result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()