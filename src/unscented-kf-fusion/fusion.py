# Basic modules
import numpy as np
import cv2
import time
from datetime import datetime
from threading import Thread

# Camera and detection modules
import pyrealsense2 as rs
from cvzone.FPS import FPS
from cvzone_fork import HandDetector

# Adjusted mathematical and realsense function
import helper_functions.functions_cv2 as fcv2
import helper_functions.functions_math as fcm

# Kalman Filter modules
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import (
    UnscentedKalmanFilter,
    MerweScaledSigmaPoints,
    JulierSigmaPoints,
)


# # # # # # # # # # # # # # # # # # # #
# Initialize basic counter constants  #
# # # # # # # # # # # # # # # # # # # #
# Time managment
pTime = 0
cTime = 0
num = 0

# Comparison of coordinates - Top view/Side view
i = 0
j = 0

# Dead pixels
c_dead_top = 0
c_dead_side = 0

# material distance
isInside = False
distance_material = 0

# # # # # # # # # #
# Hand detection  #
lm_range = range(21)
depth_distance_lst = []

# Landmark constants
dist_init_lst = []
dist_current_lst = []
lm_rangeT = range(20)
lm_rangeComparison = range(14)
lm_coord_lst = []
boxW_fix = None
boxH_fix = None
mean_depth_lms = 0
lms_shortened = True
baseline_shortened = True
occlusion_range = range(13)
lm_counter = 0

# # # # # # # # # # # # # #
# Unscented Kalman Filter #
# # # # # # # # # # # # # #

# For extensive documentation check the filter-py
# https://filterpy.readthedocs.io/en/latest/

# time-step
dt = 0.1
eps_max = 4.0
epss = []
counter = 0
Q_scale_factor = 100

# f-, h-functions

# # Initial state transition function
# def f_state_transition(x):
#     """state transition function for a constant velocity
#     with state vector [x, x', x'', y, ..]'"""

#     F = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
#     return np.dot(F, x)
#     # return F @ x

# Adjusted transitional function using expected movement
def f_state_transition(x, x_1):
    """state transition function for a constant velocity
    with state vector [x, x', x'', y, ..]'"""

    dt = 0.1
    P = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])

    w = np.array([np.random.normal(0.0, 0.04, 3)])
    A = np.exp(-2 * 0.5 * dt) * x_1
    B = 2 * np.exp(-0.5 * dt) * np.cos(2 * np.pi * 1 * dt) * x
    C = np.sqrt(P) * w

    return A + B + C


def h_observation(x):
    """state transition function for a constant velocity
    with state vector [x, x', x'', y, ..]'"""

    return np.array([[x[0]], [x[1]], [x[2]]])
    # return H @ x


# create sigma points
points_0 = MerweScaledSigmaPoints(n=3, alpha=0.1, beta=2.0, kappa=-1)
# points = JulierSigmaPoints(n=9, kappa=3.-3, sqrt_method=np.linalg.cholesky)
# points_0 = JulierSigmaPoints(n=3, kappa=3.-3)

# Create Unscented_KF class
Unscented_KF = UnscentedKalmanFilter(
    dim_x=3, dim_z=3, dt=dt, fx=f_state_transition, hx=h_observation, points=points_0
)

# x: system state state - initial position
Unscented_KF.x = np.array([[0.0], [0.0], [0.0]])  # x position  # y  # z
Unscented_KF.x_1 = np.array([[0.0], [0.0], [0.0]])

# P: estimate uncertainty matrix
Unscented_KF.P = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])

# R: measurement covariance matrix
Unscented_KF.R = 5

# Q: process noise matrix
Unscented_KF.Q = Q_discrete_white_noise(2, dt, 0.1)

try:
    # Create pipeline
    pipeline = rs.pipeline()
    pipeline2 = rs.pipeline()

    # Create a config object
    config = rs.config()
    config2 = rs.config()

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config2.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config2.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

    # start pipeline
    pipeline2.start(config2)

    # Configure the pipeline to stream the depth stream
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

    # start pipeline
    cv2.waitKey(300)
    pipeline.start(config)

    # Create colorizer object
    colorizer = rs.colorizer()

    fpsReader = FPS()
    detector = HandDetector(detectionCon=0.8, maxHands=2)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # #
    # Camera calibration  #
    # # # # # # # # # # # #
    # rotation from side to top
    rot_rodrigues_vec_21 = np.array([[0.6568], [-0.6162], [1.4306]])
    rotM_21, _ = cv2.Rodrigues(rot_rodrigues_vec_21)
    transV_21 = np.array([[0.0979416], [0.3873274], [0.2284719]])

    # Streaming loop
    while True:
        # Get frame set
        frames2 = pipeline2.wait_for_frames()
        # Get depth frame
        depth_frame2 = frames2.get_depth_frame()
        color_frame2 = frames2.get_color_frame()

        # get intrinsics of d415 - top-view
        depth_intrinsic2 = depth_frame2.profile.as_video_stream_profile().intrinsics
        color_intrinsic2 = color_frame2.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrinsic2 = depth_frame2.profile.get_extrinsics_to(
            color_frame2.profile
        )

        # Convert depth_frame to numpy array to render image in opencv
        color_image2 = np.asanyarray(color_frame2.get_data())
        align2 = rs.align(rs.stream.color)
        frames2 = align2.process(frames2)
        fps2, color_image2 = fpsReader.update(color_image2)
        color_image2 = cv2.cvtColor(color_image2, cv2.COLOR_BGR2RGB)
        hands2, color_image2 = detector.findHands(color_image2)

        # aligned depth frame - side view
        aligned_depth_frame2 = frames2.get_depth_frame()

        # Get frames
        frames = pipeline.wait_for_frames()
        # Get depth frame
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # get intrinsics of d415 - top-view
        depth_intrinsic = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrinsic = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrinsic = depth_frame.profile.get_extrinsics_to(
            color_frame.profile
        )

        # Convert depth_frame to numpy array to render image in opencv
        color_image = np.asanyarray(color_frame.get_data())
        align = rs.align(rs.stream.color)
        frames = align.process(frames)
        fps, color_image = fpsReader.update(color_image)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        hands, color_image = detector.findHands(color_image)

        # aligned depth frame - top view
        aligned_depth_frame = frames.get_depth_frame()

        # # # # # # # # # # # # # # # # # # # #
        # Drawn rectangle for the covered saw #
        # # # # # # # # # # # # # # # # # # # #
        rect_cover = cv2.rectangle(
            color_image, (5, 260), (1280 - 5, 430), (0, 255, 255), 1
        )
        cv2.putText(
            color_image,
            "cover",
            (5, 260 - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (0, 255, 255),
            1,
        )
        cover_pts = np.array(
            [[5, 260], [1280 - 5, 260], [1280 - 5, 430], [5, 430]], np.int32
        )
        cover_pts = cover_pts.reshape((-1, 1, 2))

        # rectangle saw
        rect_saw = cv2.rectangle(color_image, (220, 330), (1080, 360), (0, 0, 255), 1)
        cv2.putText(
            color_image,
            "saw",
            (220, 330 - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (0, 0, 255),
            1,
        )
        saw_pts = np.array([[220, 330], [1080, 330], [1080, 360], [220, 360]], np.int32)
        saw_pts = saw_pts.reshape((-1, 1, 2))

        # Top view
        if i == 0:
            if len(hands) != 0:
                image_points = [hands[0]["lmList"][4][0], hands[0]["lmList"][4][1]]
                coordinates_3D = fcm.convert_depth_pixel_to_metric_coordinate(
                    aligned_depth_frame,
                    image_points[0],
                    image_points[1],
                    color_intrinsic,
                )
                if coordinates_3D[2] == 0 and c_dead_top <= 5:
                    image_points_xr = image_points[0] + 1
                    coordinates_3D = fcm.convert_depth_pixel_to_metric_coordinate(
                        aligned_depth_frame,
                        image_points_xr,
                        image_points[1],
                        color_intrinsic,
                    )
                    c_dead_top += 1
                    if coordinates_3D[2] != 0:
                        c_dead_top = 6
                        print("Top view: Dead pixel adjusted!")

                # Set the expected distance to the material
                distance_material = 0.679

                # get average lm depth
                for lm in lm_range:
                    try:
                        lm_coord_lst.append(hands[0]["lmList"][lm])
                        depth_lm = aligned_depth_frame.get_distance(
                            hands[0]["lmList"][lm][0], hands[0]["lmList"][lm][1]
                        )
                        if depth_lm != 0:
                            depth_distance_lst.append(depth_lm)

                        if lm == (lm_range[-1]):
                            mean_depth_lms = sum(depth_distance_lst) / len(
                                depth_distance_lst
                            )
                            # print('depth_distance_list:\n', depth_distance_lst)
                            print("average depth of all lms:", mean_depth_lms)
                            depth_distance_lst.clear()
                    except RuntimeError:
                        pass

                # # # # # # # # # # # # # # # # # #
                # check if hand on material/desk  #
                # # # # # # # # # # # # # # # # # #

                checkInside_lst = []
                if (distance_material - 0.035) <= mean_depth_lms <= distance_material:
                    for lm in lm_range:
                        check_rectangle = cv2.pointPolygonTest(
                            cover_pts, lm_coord_lst[lm], False
                        )
                        if check_rectangle == 1:
                            print(lm, "-lm INSIDE of cover rectangle")
                        elif check_rectangle == -1:
                            print(lm, "-lm NOT in cover rectangle")
                            checkInside_lst.append([int(check_rectangle)])
                            if (
                                all(flag == [-1] for (flag) in checkInside_lst)
                                and len(checkInside_lst) == 21
                            ):
                                # Initial expected distance while the hand sits flat on the material
                                if not dist_init_lst:
                                    for lm_index in lm_rangeT:
                                        ignored_LM_lst = [0, 4, 8, 12, 16]
                                        # print('LM_index:\n', lm_index)
                                        # print('LM_index+1:\n', lm_index+1)
                                        if lm_index in ignored_LM_lst:
                                            continue
                                        dist_init, _ = detector.findDistance(
                                            hands[0]["lmList"][lm_index],
                                            hands[0]["lmList"][lm_index + 1],
                                        )
                                        dist_init_lst.append(dist_init)

                    # find current distance between relevant landmarks
                    # [1-2, 2-3, 3-4, 5-6, 6-7, 7-8,
                    #  9-10, 10-11, 11-12, 13-14, 14-15,
                    #  15-16, 17-18, 18-19, 19-20]
                    if dist_init_lst:
                        for lm_index in lm_rangeT:
                            ignored_LM_lst = [0, 4, 8, 12, 16]
                            if lm_index in ignored_LM_lst:
                                continue
                            dist_current, _ = detector.findDistance(
                                hands[0]["lmList"][lm_index],
                                hands[0]["lmList"][lm_index + 1],
                            )
                            # cv2.circle(color_image, (int(hands[0]['lmList'][lm_index][0]), int(hands[0]['lmList'][lm_index][1])), radius=12, color=(255, 0, 255),
                            #            thickness=-1)
                            dist_current_lst.append(dist_current)

                        # check if 7-8 shortened
                        for finger in occlusion_range:
                            ignored_counter = [1, 2, 4, 5, 7, 8, 10, 11]
                            if finger in ignored_counter:
                                continue
                            # check if 7-8 shortened
                            if dist_current_lst[finger + 2] <= (
                                dist_init_lst[finger + 2] * 0.95
                            ):
                                # check if 6-7 shortened
                                if dist_current_lst[finger + 1] <= (
                                    dist_init_lst[finger + 1] * 0.95
                                ):
                                    # check if 5-6 shortened: no prediction possible
                                    if dist_current_lst[finger] <= (
                                        dist_init_lst[finger] * 0.95
                                    ):
                                        pass

                                    # correct 7
                                    detector.predictLandmark(
                                        dist_init_lst[finger + 1],
                                        dist_current_lst[finger],
                                        (hands[0]["lmList"][finger + 2 + lm_counter]),
                                        (hands[0]["lmList"][finger + 3 + lm_counter]),
                                        color_image,
                                    )
                                    # correct 8
                                    detector.predictLandmark(
                                        (
                                            dist_init_lst[finger + 1]
                                            + dist_init_lst[finger + 2]
                                        ),
                                        dist_current_lst[3],
                                        (hands[0]["lmList"][finger + 2 + lm_counter]),
                                        (hands[0]["lmList"][finger + 3 + lm_counter]),
                                        color_image,
                                    )
                                    baseline_shortened = False
                                    lm_counter += 1
                                if baseline_shortened:
                                    # correct 8
                                    detector.predictLandmark(
                                        dist_init_lst[finger + 2],
                                        dist_current_lst[finger + 1],
                                        (hands[0]["lmList"][finger + 3 + lm_counter]),
                                        (hands[0]["lmList"][finger + 4 + lm_counter]),
                                        color_image,
                                    )
                                    lm_counter += 1
                                baseline_shortened = True

                    lm_coord_lst.clear()
                    dist_current_lst.clear()
                    lm_counter = 0

                # coordinates as np.array
                coordinates_vector = np.array(
                    [[coordinates_3D[0]], [coordinates_3D[1]], [coordinates_3D[2]]]
                )

                # check if current coordinates equal origin
                if coordinates_3D[2] == 0:
                    break
                i = 1

        # Side view
        if j == 0:
            if len(hands2) != 0:
                try:
                    image_points2 = [
                        hands2[0]["lmList"][4][0],
                        hands2[0]["lmList"][4][1],
                    ]
                    coordinates_3D_2 = fcm.convert_depth_pixel_to_metric_coordinate(
                        aligned_depth_frame2,
                        image_points2[0],
                        image_points2[1],
                        color_intrinsic2,
                    )
                    # check for missing depth
                    if coordinates_3D_2[2] == 0 and c_dead_side <= 5:
                        image_points2_xr = image_points2[0] + 1
                        coordinates_3D_2 = fcm.convert_depth_pixel_to_metric_coordinate(
                            aligned_depth_frame,
                            image_points2_xr,
                            image_points2[1],
                            color_intrinsic,
                        )
                        c_dead_side += 1
                        if coordinates_3D_2[2] != 0:
                            c_dead_side = 6
                            print("side view: dead pixel adjusted")
                            # break

                    # coordinates as np.array
                    coordinates_vector_2 = np.array(
                        [
                            [coordinates_3D_2[0]],
                            [coordinates_3D_2[1]],
                            [coordinates_3D_2[2]],
                        ]
                    )
                    coordinates_21 = np.add(
                        np.matmul(rotM_21, coordinates_vector_2), transV_21
                    )

                    # check if current coordinates equal origin
                    if coordinates_3D_2[2] == 0:
                        break
                    j = 1
                except:
                    pass

        # Apply the sequential Unscented_KF (predict, update, ...)
        # for the current landmark

        Unscented_KF.predict()
        Unscented_KF.update(coordinates_3D)
        Unscented_KF.predict()
        Unscented_KF.update(coordinates_3D_2)

        # Adjust position at t-1 to the last expected spot
        Unscented_KF.x_1 = Unscented_KF.x

        # calculate residual
        y, S = Unscented_KF.y, Unscented_KF.S
        eps = y.T @ np.linalg.inv(S) @ y
        epss.append(eps)
        if eps > eps_max:
            Unscented_KF.Q *= Q_scale_factor
            counter += 1
        elif counter > 0:
            Unscented_KF.Q /= Q_scale_factor
            counter -= 1

        i = j = 0

        # Resize image for visualization
        color_image_re = fcv2.image_resize(color_image, height=480)
        # Resize image for visualization
        color_image2_re = fcv2.image_resize(color_image2, height=480)

        # Render image in opencv window
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Actual windows - remove comments as needed
        cv2.imshow("Color Stream Side", color_image2_re)
        # cv2.putText(color_image2, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (22, 0, 255), 2)  # fps-Anzeige
        # cv2.imshow("Depth Stream Side", aligned_depth_frame2)
        # color_image2_re = cv2.resize(color_image2, (720, 1280))

        # Actual windows - remove comments as needed
        cv2.imshow("Color Stream Top", color_image_re)
        # cv2.putText(color_image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (22, 0, 255), 2)  # fps-Anzeige
        # cv2.imshow("Depth Stream", aligned_depth_frame)

        k = cv2.waitKey(1)

        if k == 27:
            cv2.destroyAllWindows()
            break

finally:
    pass
