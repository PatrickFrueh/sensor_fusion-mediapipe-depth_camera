import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
import cv2
from filterpy.kalman import (
    UnscentedKalmanFilter,
    MerweScaledSigmaPoints,
    JulierSigmaPoints,
)
import scipy
from scipy import stats
from numpy import linalg as la
from numpy.random import uniform
from numpy.random import seed

# # # # # # # # #
# kalman filter #
# # # # # # # # #

dt = 0.1

# # # # # # #
# linear KF #
# # # # # # #

basic_kf = KalmanFilter(dim_x=9, dim_z=3)


# x: system state state - initial position
# basic_kf.x = np.array([[0.],  # position 0, 3, 6
#                        [0.],  # velocity 1, 4, 7
#                        [0.],  # acc.     2, 5, 8
#                        [0.],
#                        [0.],
#                        [0.],
#                        [0.],
#                        [0.],
#                        [0.]])


# F: extrapolated state transition matrix
basic_kf.F = np.array(
    [
        [1.0, dt, 0.5 * dt**2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, dt, 0.5 * dt**2, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, dt, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, dt, 0.5 * dt**2],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, dt],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
)

# P: estimate uncertainty matrix
basic_kf.P = np.array(
    [
        [1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0],
    ]
)


# Q: process noise matrix
basic_kf.Q = Q_discrete_white_noise(3, dt, 0.1, block_size=3)  # process uncertainty

# H: observation matrix
basic_kf.H = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ]
)

# R: measurement covariance matrix
basic_kf.R = np.array([[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]])

# # # # # # # # #
# unscented KF  #
# # # # # # # # #


def f_state_transition(x):
    """state transition function for a constant velocity
    aircraft with state vector [x, x', x'', y, ..]'"""

    F = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return np.dot(F, x)
    # return F @ x


def h_observation(x):
    """state transition function for a constant velocity
    aircraft with state vector [x, x', x'', y, ..]'"""

    # H = np.array([[1., 0., 0.],
    #               [0., 1., 0.],
    #               [0., 0., 1.]], dtype=float)
    # return np.dot(H, x)
    return np.array([[x[0]], [x[1]], [x[2]]])
    # return H @ x


# create sigma points
points_0 = MerweScaledSigmaPoints(n=3, alpha=0.1, beta=2.0, kappa=-1)
# points = JulierSigmaPoints(n=9, kappa=3.-3, sqrt_method=np.linalg.cholesky)
# points_0 = JulierSigmaPoints(n=3, kappa=3.-3)

# Create UKF class
basic_ukf_0 = UnscentedKalmanFilter(
    dim_x=3, dim_z=3, dt=dt, fx=f_state_transition, hx=h_observation, points=points_0
)

# x: system state state - initial position
basic_ukf_0.x = np.array([[0.7], [0.0], [0.0]])  # x position  # y  # z

# P: estimate uncertainty matrix
basic_ukf_0.P = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])

# R: measurement covariance matrix
basic_ukf_0.R = 5

# Q: process noise matrix
basic_ukf_0.Q = Q_discrete_white_noise(2, dt, 0.1)

# # # # # # # # #
# unscented KF  #
# # # # # # # # #

# # # # # # # # #
# Visualization #
# # # # # # # # #

# Initialize visualization graph
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_xlabel("x-axis"), ax.set_ylabel("y-axis"), ax.set_zlabel("z-axis")

# Initialize variables such as counters, transformation matrices, ..
# Transformational matrix: calibration using rodrigues vector
i = 0
rotVec_12 = np.array([[0.5613], [0.0060], [-0.0842]])
rotM_12, _ = cv2.Rodrigues(rotVec_12)
transV_12 = np.array([[0.0610116], [0.2826146], [0.2280501]])
rotM_21 = np.linalg.inv(rotM_12)
transV_21 = -(np.matmul(rotM_21, transV_12))

# epsilon max
eps_max = 4.0
eps_lst = []
counter = 0

# receive coordinates of landmark 8 - plot
with open("landmark_8.txt") as d:
    ground_truth_x, ground_truth_y, ground_truth_z = [], [], []
    measurements_top_x, measurements_top_y, measurements_top_z = [], [], []
    measurements_side_x, measurements_side_y, measurements_side_z = [], [], []
    UKF_x, UKF_y, UKF_z = [], [], []
    for line in d:
        # get measurements - top view
        row = line.split()

        # define ground truth as initial recording (top view)
        ground_truth = np.array([float(row[0]), float(row[1]), float(row[2])])
        ground_truth_x.append(ground_truth[0])
        ground_truth_y.append(ground_truth[1])
        ground_truth_z.append(ground_truth[2])

        # simulate noisy top measurements
        measurement_top = np.array([float(row[0]), float(row[1]), float(row[2])])
        measurement_top[0] += np.random.uniform(-0.005, 0.005, 1)
        measurement_top[1] += np.random.uniform(-0.005, 0.005, 1)
        measurement_top[2] += np.random.uniform(-0.0025, 0.0025, 1)
        measurements_top_x.append(measurement_top[0])
        measurements_top_y.append(measurement_top[1])
        measurements_top_z.append(measurement_top[2])

        # transform ground truth (top view -> side view)
        # simulate noisy side measurements
        ground_truth_vector = np.array(
            [[float(row[0])], [float(row[1])], [float(row[2])]]
        )
        measurement_side = np.add(np.matmul(rotM_12, ground_truth_vector), transV_12)
        measurement_side[0] += np.random.uniform(-0.015, 0.015, 1)
        measurement_side[1] += np.random.uniform(-0.015, 0.015, 1)
        measurement_side[2] += np.random.uniform(-0.015, 0.015, 1)

        # transform side view back to top view
        measurement_side = np.add(np.matmul(rotM_21, measurement_side), transV_21)
        measurement_side = np.hstack(
            (measurement_side[0], measurement_side[1], measurement_side[2])
        )
        measurements_side_x.append(measurement_side[0])
        measurements_side_y.append(measurement_side[1])
        measurements_side_z.append(measurement_side[2])

        # # calculate residual
        # y, S = basic_ukf.y, basic_ukf.S
        # eps = y.T @ np.linalg.inv(S) @ y
        # eps_lst.append(eps * 1000)

        # plot data
        # ax.scatter(ground_truth[0], ground_truth[1], ground_truth[2], marker='o', color='black', s=4, label='Ground Truth')
        ax.scatter(
            measurement_top[0],
            measurement_top[1],
            measurement_top[2],
            marker="o",
            color="royalblue",
            s=4,
            label="Measurement Top",
        )

        ax.scatter(
            measurement_side[0],
            measurement_side[1],
            measurement_side[2],
            marker="o",
            color="slateblue",
            s=4,
            label="Measurement Side",
        )

        ax.plot(
            ground_truth_x,
            ground_truth_y,
            ground_truth_z,
            color="black",
            label="Ground Truth",
        )

        # ax.plot(measurements_top_x, measurements_top_y, measurements_top_z, color='royalblue', label='Measurement Top')
        # ax.plot(measurements_side_x, measurements_side_y, measurements_side_z, color='slateblue', label='Measurement Side')
        # ax.plot(UKF_x, UKF_y, UKF_z, color='green', label='UKF')
        # ax.scatter(basic_kf.x[0], basic_kf.x[3], basic_kf.x[6], marker='o', color='red', s=4)

        # In case an animated plot is needed
        # -> Wait for data to plot
        # plt.pause(0.03333333333)
        i += 1
        if i == 137:
            # plt.show()
            t = np.arange(0, len(eps_lst) * dt, dt)
            # plt.plot(t, eps_lst)

            # MatPlotLib specific legends/labels/titles
            plt.title("Index Finger: Movement")

            # remove duplicate labels
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc="upper center")

            # # show figure
            # plt.show()
            # # press q to close figure
