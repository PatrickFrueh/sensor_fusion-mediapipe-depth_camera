![Sensor fusion of multiple depth cameras using kalman filters](https://github.com/PatrickFrueh/sensor_fusion-mediapipe-depth_camera/blob/main/res/Fraunhofer-logo.png)

## Table of Contents

- [Description](#description)
- [Requirements](#prerequisites)
- [Installation](#installation)
- [License](#license)
- [Contact](#contact)

## Description
Hand detection using Google MediaPipe and multiple depth (stereovision) cameras.
Alignment of point clouds; fusion of multiple landmarks using Unscented Kalman Filter.

## Prerequisites
* pip

## Installation
After cloning the repo, installing the requirements should suffice.
* `git clone git@github.com:PatrickFrueh/sensor_fusion-mediapipe-depth_camera.git`
* `pip install -r requirements.txt`__
Depending on your OS pyrealsense2 might need a different approach for the installing process.

## Feature Overview
- Under `src/calibration` the different methods for stereo camera calibration are displayed.
It is suggested to use the `3d/point_clouds` approaches, as these have repeatedly shown the best results.
The resulting *homogenous transformation matrix* is used in the following steps.
- Under `src/unscented-kf-fusion` an adjusted `cvzone_fork.py` is used for the hand detection. It is used for the final fusion in `fusion.py` and `fusion-visualization.py`. 

## License
Stanard MIT license

## Contact

Surname | Name | Mail
--- | --- | ---
Frueh | Patrick | patrick.frueh@gmx.net
Brander | Tim | tim.brander@ipa.fraunhofer.de