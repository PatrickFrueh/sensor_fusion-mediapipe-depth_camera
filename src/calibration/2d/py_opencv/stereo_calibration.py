import cv2
import numpy as np
import cv2 as cv
import glob


# Size of the taken images
width = 1280
height = 720

# # Chessboard specifics
# Measured chessboard size in mm
size_of_chessboard_squares_mm = 0.0255
chessboardSize = (9, 7)
frameSize = (width, height)

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Prepare object points of the structure (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboardSize[0], 0 : chessboardSize[1]].T.reshape(-1, 2)
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpointsL = []  # 2d points in image plane.
imgpointsR = []  # 2d points in image plane.


imagesLeft = sorted(glob.glob("../calibration_images/one/*.png"))
imagesRight = sorted(glob.glob("../calibration_images/two/*.png"))

for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)
    print(retR)
    # If found, add object points, image points (after refining them)
    if retL and retR:

        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        print(imgLeft)
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow("img left", imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow("img right", imgR)
        cv.waitKey(10000)


cv.destroyAllWindows()

# # Set intrinsic parameters of matrices D455
# One - factory setting
newCameraMatrixR = np.array(
    [[420.739, 0.0, 432.947], [0.0, 420.401, 240.798], [0.0, 0.0, 1.0]]
)
distR = np.array([[-0.0564105, 0.0731941, -0.000467269, -0.000279257, -0.0238872]])

# Side - factory setting
newCameraMatrixL = np.array(
    [[426.478, 0.0, 431.973], [0.0, 426.011, 241.741], [0.0, 0.0, 1.0]]
)
distL = np.array([[-0.0536413, 0.0660728, -0.000184013, 0.00114153, -0.0217815]])

# Set necessary flags
flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Calculate transformation between the two cameras and
# calculate essential and fundamental matrix
(
    retStereo,
    newCameraMatrixL,
    distL,
    newCameraMatrixR,
    distR,
    rot,
    trans,
    essentialMatrix,
    fundamentalMatrix,
) = cv.stereoCalibrate(
    objpoints,
    imgpointsL,
    imgpointsR,
    newCameraMatrixL,
    distL,
    newCameraMatrixR,
    distR,
    grayL.shape[::-1],
    criteria_stereo,
    flags,
)

print("retStereo, reprojection error\n", retStereo)
print("newCameraMatrixL\n", newCameraMatrixL)
print("distL\n", distL)
print("newCameraMatrixR\n", newCameraMatrixR)
print("distR\n", distR)
print("essentialMatrix\n", essentialMatrix)
print("fundamentalMatrix\n", fundamentalMatrix)

# Relevant 3x3 Rotational Matrix and relevant 3x1 Translational Vector
# Spanning 4x4 Transformation Matrix
print("Rotational Matrix:\n", rot)
print("Translational Vector:\n", trans)
