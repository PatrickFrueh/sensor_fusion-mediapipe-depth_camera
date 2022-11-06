# # Take a snapshot of the chessboard setting using the 's'-button
# # Ideally image pairs of > 15 should be taken for decent results

# importing relevant libraries
import pyrealsense2 as rs
import numpy as np
import cv2
import logging
import time

pTime = 0
cTime = 0
num = 0

# set parameters
width = 1280
height = 720
fps = 30

# Configure depth and color streams...
# ...of Camera 1
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device("151422254978")
config_1.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
config_1.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

# ...of Camera 2
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device("151422254651")
config_2.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
config_2.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

# Start streaming from both cameras
pipeline_profile_1 = pipeline_1.start(config_1)
pipeline_profile_2 = pipeline_2.start(config_2)

# Activate realsense colorizer
colorizer = rs.colorizer()

try:
    while True:

        # Camera 1
        # Wait for a coherent pair of frames: depth and color
        frames_1 = pipeline_1.wait_for_frames()
        depth_frame_1 = frames_1.get_depth_frame()
        color_frame_1 = frames_1.get_color_frame()

        # Camera intrinsics of camera 1 -> adjust for intrinsics of camera 2
        color_intrinsic_1 = color_frame_1.profile.as_video_stream_profile().intrinsics
        depth_intrinsic_1 = depth_frame_1.profile.as_video_stream_profile().intrinsics

        if not depth_frame_1 or not color_frame_1:
            continue

        # Convert images to numpy arrays
        depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())

        # Camera 2
        # Wait for a coherent pair of frames: depth and color
        frames_2 = pipeline_2.wait_for_frames()
        depth_frame_2 = frames_2.get_depth_frame()
        color_frame_2 = frames_2.get_color_frame()

        if not depth_frame_2 or not color_frame_2:
            continue

        # Convert images to numpy arrays
        depth_image_2 = np.asanyarray(depth_frame_2.get_data())
        color_image_2 = np.asanyarray(color_frame_2.get_data())

        # cv2.imshow("Depth Stream", depth_color_image)
        cv2.imshow("Color Stream Top", color_image_2)
        # cv2.imshow("Depth Stream Top", depth_colormap_2)

        # cv2.imshow("Depth Stream Side", depth_color_image2)
        # color_image2_re = cv2.resize(color_image2, (720, 1280))
        cv2.imshow("Color Stream Side", color_image_1)

        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break
        elif k == ord("s"):  # wait for 's' key to save and exit
            cv2.imwrite(
                "calibration_images/image_top_" + str(num) + ".png", color_image_2
            )
            cv2.imwrite(
                "calibration_images/image_side_" + str(num) + ".png", color_image_1
            )
            print("Image pair saved.")
            num += 1

finally:
    pass
