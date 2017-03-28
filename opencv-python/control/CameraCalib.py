import numpy as np
import cv2
import glob
from .util_fxn import intersection

class CameraCalib:
    def __init__(self, src, src_id, dim, target, callback):
        self.__sources = src
        self.__src = src.get(src_id)
        self.target = target
        self.frame_count = 0

        self.dim = dim
        self.size = self.__src.size()
        self.objp = np.zeros((dim[0]*dim[1],3), np.float32)
        self.objp[:,:2] = np.mgrid[0:dim[1], 0:dim[0]].T.reshape(-1,2)

        self.objectPoints = []
        self.imagePoints = [[], []]

        self.flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        self.criteria = criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.calib_done = False
        self.cb = callback
    
    def is_ready(self):
        return self.frame_count == self.target
    
    def update(self):
        if self.is_ready():
            return
        found = False
        frame_l, frame_r = self.__src.frames()
        frame_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        frame_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        corners_l = None
        corners_r = None
        for scale in [1, 2]:
            if scale == 1:
                img_l = frame_l
                img_r = frame_r
            else:
                img_l = cv2.resize(frame_l, (0,0), fx=scale, fy=scale)
            ret_l, corners_l = cv2.findChessboardCorners(frame_l, self.dim, self.flags)
            ret_r, corners_r = cv2.findChessboardCorners(frame_r, self.dim, self.flags)
            found = ret_l and ret_r
            if found:
                if scale > 1:
                    corners_l *= 1.0 / scale
                    corners_r *= 1.0 / scale
                    print('SC')
                break
        if found:
            self.objectPoints.append(self.objp)
            corners2l = cv2.cornerSubPix(frame_l, corners_l, (11, 11), (-1, -1), self.criteria)
            corners2r = cv2.cornerSubPix(frame_r, corners_r, (11, 11), (-1, -1), self.criteria)
            self.imagePoints[0].append(corners2l)
            self.imagePoints[1].append(corners2r)
            self.frame_count += 1
            if self.is_ready():
                self.__calib()
            return True
        return False
    
    def __calib(self):
        if not (self.is_ready() or self.calib_done):
            return
        self.objectPoints = np.array(self.objectPoints)
        self.imagePoints[0] = np.array(self.imagePoints[0])
        self.imagePoints[1] = np.array(self.imagePoints[1])
        # Get camera matrix
        cameraMatrix = [None, None]
        cameraMatrix[0] = cv2.initCameraMatrix2D(
            self.objectPoints, self.imagePoints[0], self.size, 0)
        cameraMatrix[1] = cv2.initCameraMatrix2D(
            self.objectPoints, self.imagePoints[1], self.size, 0)
        distCoeff = [np.zeros(4, np.float32), np.zeros(4, np.float32)]
        # Get camera calibration info
        rms, M1, D1, M2, D2, R, T, E, F = cv2.stereoCalibrate(
            self.objectPoints, self.imagePoints[0], self.imagePoints[1],
            cameraMatrix[0], distCoeff[0],
            cameraMatrix[1], distCoeff[1],
            self.size,
            flags = cv2.CALIB_FIX_ASPECT_RATIO +
                    cv2.CALIB_ZERO_TANGENT_DIST +
                    cv2.CALIB_USE_INTRINSIC_GUESS +
                    cv2.CALIB_SAME_FOCAL_LENGTH +
                    cv2.CALIB_RATIONAL_MODEL +
                    cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5,
            criteria=self.criteria)
        self.cameraMatrix = [M1, M2]
        self.distCoeff = [D1, D2]
        self.r = R
        self.t = T
        # Get rectification info
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(M1, D1,
            M2, D2, self.size, R, T,
            flags = cv2.CALIB_ZERO_DISPARITY, alpha = 1)
        self.r1 = R1
        self.r2 = R2
        self.p1 = P1
        self.p2 = P2
        self.q = Q
        self.validRoi = [roi1, roi2]
        # Get undistortion matrices
        self.m1_ = cv2.initUndistortRectifyMap(self.cameraMatrix[0], self.distCoeff[0], self.r1, self.p1, self.size, cv2.CV_16SC2)
        self.m2_ = cv2.initUndistortRectifyMap(self.cameraMatrix[1], self.distCoeff[1], self.r2, self.p2, self.size, cv2.CV_16SC2)
        # Intersect to get common area
        int_roi = intersection(*self.validRoi)
        # Convert to bounds
        self.roi = (int_roi[1], int_roi[1] + int_roi[3],
            int_roi[0], int_roi[0] + int_roi[2])
        self.cb()