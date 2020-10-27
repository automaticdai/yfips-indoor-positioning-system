# YF-IPS: Robot Indoor Positioning System
![](https://img.shields.io/github/stars/yfrobotics/yfips-indoor-positioning-system) ![](https://img.shields.io/github/issues/yfrobotics/yfips-indoor-positioning-system) ![](https://img.shields.io/github/license/yfrobotics/yfips-indoor-positioning-system)

## 1. Introduction
Traditional localisation system is not friendly to robot researchers and learners, both (a) they are expensive; (b) they need a lot of efforts to setup. In this project, we target to use low cost sensors to achieve an indoor positioning system (IPS) which has relatively good performance and can also be used for tracking and navigation proposes beyond robotic applications.

The system is designed in mind that it will be used to locate multiple robots (1~100) and at the same time with a high speed (at least 30 fps). After configuration and calibration, the system will return the position of the robot in the form of (id, x, y, yaw), in which id is the robot identification number, (x, y) is the 2D coordination relative to the origin point, yaw is the rotation on the z axis.

This project has two versions: (1) vision-based; (2) ToF/tag-based, which can be used together or separately. The project is currently in its very early stage. We want to explore which technology / combination of technologies could produce satisfactory precision with affordable cost. We are working on vision-based and tag-based and are evaluating if these are sufficient. 


## 2. Design specification
These are the objectives of this project. 

- The system can detect robots in a region of 5 x 5 meters.
- The precision with the simpliest setup has +-20 cm. Using more sensors could improve the precision up to +-10 cm.
- The system can simulatenously detect 1-10 robots with at least 10 fps.
- The system can send the output through UDP/IP or as a ROS topic.


## 3. Hardware Requirements
### 3.1 PC
A PC (Desktop/Laptop) is needed to run the program.

### 3.2 Camera
You need a high-resolution HD (1080p) web camera  in order to use the vision version of this code. I suggest a Logitech C920/C922 as this gives the optimal image quality for localisation propose but still with a affordable price. 

### 3.3 Vision Tags
TBD.

### 3.4 Wireless Anchors
TBD.

### 3.5 Wireless Tags
TBD.


## 4. How to install and run
(1) Set a virtual environment: 

`virtualenv -p python3 venv/`

`source venv/bin/activate`

(2) Upgrade PIP:

`pip3 install --upgrade pip`

(3) Install the requirements:

`sudo pip3 install requirements.txt`


## 5. Calibration
### 5.1 Camera calibration
To estimate tag pose, you need to know the `intrinsic camera parameters`, which can be estimated using the `calibrate_camera.py` script.
You also need to know the `tag size` in order to scale the estimated translation vectors correctly.

### 5.2 Environment calibration
You need to calibrate the environment by giving the four corner points and their distance. You also need to assign the origin point (0,0).


## 6. Credits
- This project is built using [OpenCV 4](https://opencv.org/opencv-4-0/) and [Qt 5](https://www.qt.io/). 
- The Apriltag detection and pose estimation are based on: https://github.com/swatbotics/apriltag


## 7. Contributors
- [automaticdai](https://github.com/automaticdai)
- [xinyu-xu-dev](https://github.com/xinyu-xu-dev)


## License
MIT
