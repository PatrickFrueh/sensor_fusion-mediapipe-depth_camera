import numpy as np
import cv2


# get point linearly
def linear_approx(p1, p2, length, s):
    """
    p1 -> p2: vector; therefore: p2: point to extend
    length: length between two points
    s: scale by which point should be extended
    """

    # outer_point length
    outer_p = length + s * length

    # current distance
    current_distance = length

    # relation of outer_p to current_distance
    t = outer_p / current_distance
    x, y = ((1 - t) * p1[0] + t * p2[0]), ((1 - t) * p1[1] + t * p2[1])

    return x, y


# rotate image
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


# metric coordinates in position x,y (meters) using distance
def convert_depth_pixel_to_metric_coordinate(
    depth_frame, pixel_x, pixel_y, camera_intrinsics
):
    """
    Convert the depth and image point information to metric coordinates
    Parameters:
    -----------
    depth 	 	 	 : double
                                               The depth value of the image point
    pixel_x 	  	 	 : double
                                               The x value of the image coordinate
    pixel_y 	  	 	 : double
                                                    The y value of the image coordinate
    camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
    Return:
    ----------
    X : double
            The x value in meters
    Y : double
            The y value in meters
    Z : double
            The z value in meters
    """
    depth = depth_frame.get_distance(pixel_x, pixel_y)
    # depth -= 0.02
    x = (pixel_x - camera_intrinsics.ppx) / camera_intrinsics.fx * depth
    y = (pixel_y - camera_intrinsics.ppy) / camera_intrinsics.fy * depth
    return x, y, depth


# metric coordinates in position x,y (meters) using distance
def set_depth_convert_depth_pixel_to_metric_coordinate(
    z, pixel_x, pixel_y, camera_intrinsics
):
    """
    Convert the depth and image point information to metric coordinates
    Parameters:
    -----------
    depth 	 	 	 : double
                                               The depth value of the image point
    pixel_x 	  	 	 : double
                                               The x value of the image coordinate
    pixel_y 	  	 	 : double
                                                    The y value of the image coordinate
    camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
    Return:
    ----------
    X : double
            The x value in meters
    Y : double
            The y value in meters
    Z : double
            The z value in meters
    """
    x = (pixel_x - camera_intrinsics.ppx) / camera_intrinsics.fx * z
    y = (pixel_y - camera_intrinsics.ppy) / camera_intrinsics.fy * z
    return x, y, z


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def weight_perspectives(p1, p2, w1, w2):

    global p_new

    if w1 + w2 != 1:
        if w1 == 0 and w2 == 0:
            p_new = 0
        if w1 > w2:
            w2 = 1 - w1
        elif w2 < w1:
            w1 = 1 - w2

    # check if w1 + w2 == 1

    if w1 + w2 == 1:
        p_new = p1 * w1 + p2 * w2

    if w1 + w2 == 2:
        p_new = p1 * 0.5 + p2 * 0.5

    return p_new


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # #
# add coordinate origin #
# # # # # # # # # # # # #

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def cuboid_data(center, size):
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = np.array(
        [
            [
                o[0],
                o[0] + l,
                o[0] + l,
                o[0],
                o[0],
            ],  # x coordinate of points in bottom surface
            [
                o[0],
                o[0] + l,
                o[0] + l,
                o[0],
                o[0],
            ],  # x coordinate of points in upper surface
            [
                o[0],
                o[0] + l,
                o[0] + l,
                o[0],
                o[0],
            ],  # x coordinate of points in outside surface
            [o[0], o[0] + l, o[0] + l, o[0], o[0]],
        ]
    )  # x coordinate of points in inside surface
    y = np.array(
        [
            [
                o[1],
                o[1],
                o[1] + w,
                o[1] + w,
                o[1],
            ],  # y coordinate of points in bottom surface
            [
                o[1],
                o[1],
                o[1] + w,
                o[1] + w,
                o[1],
            ],  # y coordinate of points in upper surface
            [o[1], o[1], o[1], o[1], o[1]],  # y coordinate of points in outside surface
            [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w],
        ]
    )  # y coordinate of points in inside surface
    z = np.array(
        [
            [o[2], o[2], o[2], o[2], o[2]],  # z coordinate of points in bottom surface
            [
                o[2] + h,
                o[2] + h,
                o[2] + h,
                o[2] + h,
                o[2] + h,
            ],  # z coordinate of points in upper surface
            [
                o[2],
                o[2],
                o[2] + h,
                o[2] + h,
                o[2],
            ],  # z coordinate of points in outside surface
            [o[2], o[2], o[2] + h, o[2] + h, o[2]],
        ]
    )  # z coordinate of points in inside surface
    return x, y, z


if __name__ == "__main__":
    center = [0.075, 0, 0.75]
    length = 0.05
    width = 0.8
    height = 0.7
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection="3d")
    X, Y, Z = cuboid_data(center, (length, width, height))
    ax1.plot_surface(X, Y, Z, color="gold", rstride=1, cstride=1, alpha=0.25)
    ax1.set_xlabel("X")
    ax1.set_xlim(-0.05, 0.2)
    ax1.set_ylabel("Y")
    ax1.set_ylim(-0.4, 0.4)
    ax1.set_zlabel("Z")
    ax1.set_zlim(0.4, 1.1)
    plt.show()
