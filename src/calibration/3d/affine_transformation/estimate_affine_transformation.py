# This script uses a similar approach to the MatLab stereo camera calibrator
# Firstly, the chessboard needs to be detected in each camera.
# Once a chessboard is detected -> pressing the 'b'-button saves the coordinates
# After re-placing the chessboard somewhere in the room: the 'n'-button needs to be pressed.
# Save coordinates using 'b'
# Repeat...

# Lastly: After gathering sufficient 3d-coordinates: press the 't'-button to estimate the affine transformation
# ..using the opencv-function estimateAffine3D

import pyrealsense2 as rs
import numpy as np
import cv2


pTime = 0
cTime = 0
num = 0
posList = []
posList2 = []
# cv2.estimateTranslation3D()
# initialize shared coordinate system lists
coordinates_3d_top_view_lst = []
coordinates_3d_side_view_lst = []

# Configure depth and color streams...
# width, height
width = 1280
height = 720
fps = 30
alpha = 0.5

# misc variables
test_set_ready = True

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

# Activate RealSense colorizer
colorizer = rs.colorizer()


# Chessboard specifics
chessboardSize = (9, 7)
frameSize = (width, height)
size_of_chessboard_squares_mm = 25

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboardSize[0], 0 : chessboardSize[1]].T.reshape(-1, 2)
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpointsL = []  # 2d points in image plane.
imgpointsR = []  # 2d points in image plane.

# pixelpoints
pixelpoints_top_view_lst = []
pixelpoints_side_view_lst = []
delta_lst = []
shared_3d = []

alpha_0 = 0.5

# stream specific data
depth_scale_1 = pipeline_profile_1.get_device().first_depth_sensor().get_depth_scale()
depth_scale_2 = pipeline_profile_2.get_device().first_depth_sensor().get_depth_scale()


try:
    while True:
        frames_1 = pipeline_1.wait_for_frames()
        depth_frame_1 = frames_1.get_depth_frame()
        color_frame_1 = frames_1.get_color_frame()

        # Intrinsic and extrinsic matrices of camera 1
        color_intrinsic_1 = (
            color_frame_1.profile.as_video_stream_profile().intrinsics
        )  #
        depth_intrinsic_1 = (
            depth_frame_1.profile.as_video_stream_profile().intrinsics
        )  #
        depth_to_color_extrinsic_1 = depth_frame_1.profile.get_extrinsics_to(
            color_frame_1.profile
        )
        color_to_depth_extrinsic_1 = color_frame_1.profile.get_extrinsics_to(
            depth_frame_1.profile
        )

        if not depth_frame_1 or not color_frame_1:
            continue

        # Convert images to numpy arrays
        depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        color_image_1 = cv2.cvtColor(color_image_1, cv2.COLOR_BGR2RGB)

        # align depth_to_rgb 1
        align_1 = rs.align(rs.stream.color)
        frames_1 = align_1.process(frames_1)
        aligned_depth_frame_1 = frames_1.get_depth_frame()
        colorized_depth_1 = np.asanyarray(
            colorizer.colorize(aligned_depth_frame_1).get_data()
        )

        # Camera 2
        # Wait for a coherent pair of frames: depth and color
        frames_2 = pipeline_2.wait_for_frames()
        depth_frame_2 = frames_2.get_depth_frame()
        color_frame_2 = frames_2.get_color_frame()

        color_intrinsic_2 = (
            color_frame_2.profile.as_video_stream_profile().intrinsics
        )  #
        depth_intrinsic_2 = (
            depth_frame_2.profile.as_video_stream_profile().intrinsics
        )  #
        depth_to_color_extrinsic_2 = depth_frame_2.profile.get_extrinsics_to(
            color_frame_2.profile
        )
        color_to_depth_extrinsic_2 = color_frame_2.profile.get_extrinsics_to(
            depth_frame_2.profile
        )

        if not depth_frame_2 or not color_frame_2:
            continue

        # Convert images to numpy arrays
        depth_image_2 = np.asanyarray(depth_frame_2.get_data())
        color_image_2 = np.asanyarray(color_frame_2.get_data())

        # align depth_to_rgb 2
        align_2 = rs.align(rs.stream.color)
        frames_2 = align_2.process(frames_2)
        aligned_depth_frame_2 = frames_2.get_depth_frame()
        # depth visualization
        # colorized_depth_2 = np.asanyarray(colorizer.colorize(filtered_2).get_data())
        colorized_depth_2 = np.asanyarray(
            colorizer.colorize(aligned_depth_frame_2).get_data()
        )

        # prepare chessboard detection by graying images
        grayL = cv2.cvtColor(color_image_2, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(color_image_1, cv2.COLOR_BGR2GRAY)

        # find chessboard corners
        retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)

        # side view
        if cv2.waitKey(33) == ord("b"):
            if test_set_ready:
                for x in range(len(cornersR)):
                    rs.rs2_project_color_pixel_to_depth_pixel(
                        depth_frame_2.get_data(),
                        depth_scale_2,
                        0.2,
                        1,
                        depth_intrinsic_2,
                        color_intrinsic_2,
                        depth_to_color_extrinsic_2,
                        color_to_depth_extrinsic_2,
                        [cornersR[x][0][0], cornersR[x][0][1]],
                    )
                    coordinates_3d_top_shared = rs.rs2_deproject_pixel_to_point(
                        depth_intrinsic_2,
                        [cornersR[x][0][0], cornersR[x][0][1]],
                        depth_frame_2.get_distance(
                            int(cornersR[x][0][0]), int(cornersR[x][0][1])
                        ),
                    )
                    # coordinates_3d_top_shared = rs.rs2_deproject_pixel_to_point(color_intrinsic_2, [cornersR[x][0][0], cornersR[x][0][1]],
                    #                                                             aligned_depth_frame_2.get_distance(int(cornersR[x][0][0]), int(cornersR[x][0][1])))
                    coordinates_3d_top_view_lst.append(coordinates_3d_top_shared)
                    pixelpoints_top_view_lst.append(
                        [cornersR[x][0][0], cornersR[x][0][1]]
                    )
                # top view
                for y in range(len(cornersL)):
                    rs.rs2_project_color_pixel_to_depth_pixel(
                        depth_frame_1.get_data(),
                        depth_scale_1,
                        0.2,
                        1,
                        depth_intrinsic_1,
                        color_intrinsic_1,
                        depth_to_color_extrinsic_1,
                        color_to_depth_extrinsic_1,
                        [cornersR[y][0][0], cornersR[y][0][1]],
                    )
                    coordinates_3d_side_shared = rs.rs2_deproject_pixel_to_point(
                        depth_intrinsic_1,
                        [cornersR[y][0][0], cornersR[y][0][1]],
                        depth_frame_1.get_distance(
                            int(cornersR[y][0][0]), int(cornersR[y][0][1])
                        ),
                    )

                    coordinates_3d_side_view_lst.append(coordinates_3d_side_shared)
                    pixelpoints_side_view_lst.append(
                        [cornersL[y][0][0], cornersL[y][0][1]]
                    )

                if len(coordinates_3d_top_view_lst) >= 54:

                    test_set_ready = False
                    print("Saved 3d coordinates of corresponding points in each view.")
                    print("Change position of chessboard.")
                    print('Press "n" once ready.')

        # ready up next test_set
        if cv2.waitKey(33) == ord("n"):
            print('Getting new set of coordinates upon "b" press..')
            print("TOP-LIST:", coordinates_3d_top_view_lst)
            print("SIDE-LIST:", coordinates_3d_side_view_lst)
            test_set_ready = True

        # Clear/Empty current calibration check list on 'c' press
        if cv2.waitKey(33) == ord("c"):
            coordinates_3d_top_view_lst.clear()
            coordinates_3d_side_view_lst.clear()
            pixelpoints_side_view_lst.clear()
            pixelpoints_top_view_lst.clear()
            delta_lst.clear()
            print("..cleared lists")

        if cv2.waitKey(33) == ord("t"):
            if coordinates_3d_side_view_lst:
                coordinates_3d_top_np = np.asarray(coordinates_3d_top_view_lst)
                coordinates_3d_side_np = np.asarray(coordinates_3d_side_view_lst)
                print("TOP 3D asarray:", coordinates_3d_top_np)
                print("SIDE 3D asarray:", coordinates_3d_side_np)
                ret_val, transformation_3x4, inliers = cv2.estimateAffine3D(
                    coordinates_3d_side_np, coordinates_3d_top_np
                )
                print("Estimated 3D Affine:")
                print(transformation_3x4)

        if retL and retR:
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            imgpointsL.append(cornersL)

            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            imgpointsR.append(cornersR)

            # Draw and display the corners
            cv2.drawChessboardCorners(color_image_2, chessboardSize, cornersL, retL)
            # cv2.imshow('img top', color_image_2)
            cv2.drawChessboardCorners(color_image_1, chessboardSize, cornersR, retR)
            # cv2.imshow('img side', color_image_1)
            cv2.waitKey(1)

        cv2.imshow("Color Stream Side", color_image_2)
        cv2.imshow("Color Stream Top", color_image_1)

    # cv2.waitKey(1)
    # cv2.destroyAllWindows()

finally:
    pass
