# YF-IPS: Robot Indoor Positioning System
![](https://img.shields.io/github/stars/yfrobotics/yfips-indoor-positioning-system) ![](https://img.shields.io/github/issues/yfrobotics/yfips-indoor-positioning-system) ![](https://img.shields.io/github/license/yfrobotics/yfips-indoor-positioning-system)

## 0. Introduction
Traditional localisation system is not friendly to robot researchers and learners, both (a) they are expensive; (b) they need a lot of efforts to setup. In this project, we target to use low cost sensors to achieve an indoor positioning system (IPS) which has relatively good performance and can also be used for tracking and navigation proposes beyond robotic applications.

The system is designed in mind that it will be used to locate multiple robots (1~100) and at the same time with a high speed (at least 30 fps). After configuration and calibration, the system will return the position of the robot in the form of (id, x, y).

This project has two versions: (1) vision-based; (2) ToF/tag-based, which can be used together or separately. The project is currently in its very early stage. We want to explore which technology / combination of technologies could produce satisfactory precision with affordable cost. We are working on vision-based and tag-based and are evaluating if these are sufficient. The design specification is the system can detect a robot as far as 5 m away with a precision of +-10 cm.


## 1. Hardware
### Camera
You need a web camera in order to use the vision version of this code. I suggest a Logitech C920 as this gives the optimal image quality for localisation propose but still with a reasonable price. 

### Tags
Model to be determined.


## 2. How to install
(1) Set a virtual environment: 

`virtualenv -p python3 ./venv/`

`source ./venv/bin/activate`

(2) Install the requirements: 

`pip3 install --upgrade pip`

and then

`sudo pip3 install requirements.txt`

## 3. Credits
This project is based on [OpenCV 4](https://opencv.org/opencv-4-0/) and [Qt 5](https://www.qt.io/).


## 4. Contributors
- [automaticdai](https://github.com/automaticdai)
- [xinyu-xu-dev](https://github.com/xinyu-xu-dev)


## License
MIT
