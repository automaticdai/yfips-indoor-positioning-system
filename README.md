# yfips-indoor-positioning-system


## Introduction
Traditional localization system is not friendly to robot researchers and learners, both (a) they are expensive; (b) they need a lot of efforts to setup.

In this work, we target to use low cost sensors to achive a (relatively good) indoor positioning system (IPS) which can then be used for tracking and navigation proposes.

The system is designed in mind that it will be used to locate multiple robots (1~100) and at the same time with a high speed (at least 30fps). 

After configuration and calibration, the system will return the position of the robot in the form of (id, x, y).

This project has two versions: (1) vision-based; (2) tof/tag-based, which can be used together or seperately.


## Hardware
### Camera
You need a webcamera in order to use the vision version of this code. I suggest a Logitech C920 as this gives the optimal image quality for localization propose but still with a resonable price. 


## How to install
`virtualenv -p python3 ./venv/`

`source ./venv/bin/activate`

`which python3`

`pip3 install --upgrade pip`

`sudo pip3 install requirements.txt`


## Credits
This project is based on [OpenCV 4](https://opencv.org/opencv-4-0/) and [Qt 5](https://www.qt.io/).


## Contributors
- [automaticdai](https://github.com/automaticdai)
- [xinyu-xu-dev](https://github.com/xinyu-xu-dev)


## License
MIT
