# yf-VisionNav

Traditional localization system is expensive and not friendly to new robot researchers and learners.

In this project, we target to use low cost web cameras to achive (relatively good) positioning which can then be used for navigation propose.
The system is designed in the mind that it will be used to locate multiple robots (~100) at the same time with a high speed (at least 30fps). 

After configuration and calibration, the system will return the position of the robot in the form of (x, y).


## How to install
`virtualenv -p python3 ./venv/`

`source ./venv/bin/activate`

`which python3`

`pip3 install --upgrade pip`

`sudo pip3 install requirements.txt`


## Credits
This project is based on [OpenCV 4](https://opencv.org/opencv-4-0/) and [Qt 5](https://www.qt.io/).


## Contributors
- automaticdai


## License
MIT
