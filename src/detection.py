import numpy as np
import cv2
import apriltag
import matplotlib.pyplot as plt
import time
import json
from PIL import ImageFont, ImageDraw, Image

# configurations; will move to a config file
image_width = 640
image_height = 480
window_name = "YFIPS"


def load_config():
    pass


calib_iter = 0
calib_points = np.array([(0,0), (0,0), (0,0), (0,0)])
def mouse(event,x,y,flags,param):
    global calib_iter
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(calib_iter, (x,y))
        
        calib_points[calib_iter] = (x,y)
        calib_iter = calib_iter + 1

        if calib_iter == 4:
            calib_iter = 0
            # save parameters to json

        return True


def transform(xy):
    return (xy[0] / image_width, xy[1] / image_height)




# This will be started as a background thread
# Support tags:
# - fiducial 
# - AprilTag (ok)
# - ArUco
if __name__ == "__main__":
    # campture from the camera
    cap = cv2.VideoCapture(0)

    # set camera properties
    # 0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
    # 1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
    # 2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
    # 3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
    # 4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
    # 5. CV_CAP_PROP_FPS Frame rate.
    # 6. CV_CAP_PROP_FOURCC 4-character code of codec.
    # 7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
    # 8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
    # 9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
    # 10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
    # 11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
    # 12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
    # 13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
    # 14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
    # 15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
    # 16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
    # 17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
    # 18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
    cap.set(3, image_width)
    cap.set(4, image_height)
    cap.set(5, 60)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse)

    while(True):
        # get current time which is used for calculating FPS
        now = time.time()

        # Capture frame-by-frame
        ret, frame = cap.read()
        img = frame

        # convert into gray scale image
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # show the world corner points
        for point in calib_points:
            cv2.circle(img, (round(point[0]), round(point[1])), 3, (255, 0, 0), -1)

        # detect the apriltag
        detector = apriltag.Detector()
        result = detector.detect(img_gray)

        if result:
            cx = result[0].center[0]
            cy = result[0].center[1]

            corners = result[0].corners

            for corner in corners:
                cv2.circle(img, (round(corner[0]), round(corner[1])), 3, (0, 0, 255), -1)

            cv2.circle(img, (round(cx), round(cy)), 3, (0, 0, 255), -1)
    
            print(transform(result[0].center))

            # direction estimation


            # pose estimation


            # display the result in matplotlib
            plt.clf()
            plt.plot(cx, cy, 'ro')
            for corner in corners:
                plt.plot(round(corner[0]), round(corner[1]), 'go')
            for point in calib_points:
                plt.plot(round(point[0]), round(point[1]), 'bo')
            plt.xlim(0, image_width)
            plt.ylim(0, image_height)
            plt.pause(0.10)
            plt.show(block=False)

            #print(result[0].corners)
            #print(result[0].homography)

        # show FPS on the bottom of the screen
        fps = round(1.0 / (time.time() - now), 1)
        cv2.putText(img, "fps: {:.1f}".format(fps), (0, image_height - 10), 0, 0.4, (0,0,255))

        # Display the result frame
        cv2.imshow(window_name, img)

        # Press 'ESC' to quit:
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
