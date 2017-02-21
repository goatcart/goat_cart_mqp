import cv2
from cv2.ximgproc import createDisparityWLSFilter, createRightMatcher
import numpy as np
from enum import Enum
import json
from .util_fxn import load_mat, intersection

def fullname(o):
  return o.__class__.__module__ + "." + o.__class__.__name__

def P_base(imChannels, blockSize):
    return (8 * imChannels * blockSize * blockSize)

def calc_disp(width, scale):
    return ((int(width * scale / 8 + 0.5) + 15) & -16)

class MatcherType(Enum):
    stereo_bm   = 0
    stereo_sgbm = 1

class StereoVision:
    # Final depth map scaling factor
    factor = 1 / 16.0

    def __init__(self, src):
        self.__src = src
        self.__load_settings()
        self.__init_matcher()

    def __load_settings(self):

        fs = json.load(open('params.json', 'r'))
        i_matcher = fs['matcher']
        cam_props = fs['src_props'][i_matcher['prop']]

        self.src_id = i_matcher['src']

        if i_matcher['mode'] == 1:
            self.mode = MatcherType.stereo_sgbm
        else:
            self.mode = MatcherType.stereo_bm

        self.scale_factor = self.__src.get(self.src_id).scale()

        if self.mode == MatcherType.stereo_sgbm:
            self.block_size = 5
        elif self.scale_factor < 0.99:
            self.block_size = 7
        else:
            self.block_size = 15

        self.size                = self.__src.get(self.src_id).size()
        if i_matcher['numDisparities'] < 0:
            self.num_disp        = calc_disp(self.size[0], self.scale_factor)
        else:
            self.num_disp        = i_matcher['numDisparities']
        self.pre_filter_cap      = i_matcher['preFilterCap']
        self.uniqueness_ratio    = i_matcher['uniquenessRatio']
        self.disp_12_max_diff    = i_matcher['disp12MaxDiff']
        self.specke_range        = i_matcher['speckleRange']
        self.speckle_window_size = i_matcher['speckleWindowSize']
        self.wls_lamba           = i_matcher['wlsLambda']
        self.wls_sigma           = i_matcher['wlsSigma']

        # For rectification, load camera intrinsics + extrinsics
        self.m1                  = load_mat(cam_props['M1'])
        self.d1                  = load_mat(cam_props['D1'])
        self.m2                  = load_mat(cam_props['M2'])
        self.d2                  = load_mat(cam_props['D2'])
        self.r                   = load_mat(cam_props['R'])
        self.t                   = load_mat(cam_props['T'])
        self.r1                  = load_mat(cam_props['R1'])
        self.p1                  = load_mat(cam_props['P1'])
        self.r2                  = load_mat(cam_props['R2'])
        self.p2                  = load_mat(cam_props['P2'])
        self.q                   = load_mat(cam_props['Q'])

# TODO convert to using BM instead of SGBM

    def __init_matcher(self):
        # Create matcher
        if self.mode == MatcherType.stereo_bm:
            self.__left = cv2.StereoBM_create(self.num_disp, self.block_size)
        else:
            self.__left = cv2.StereoSGBM_create(0, self.num_disp, self.block_size)
            self.__left.setP1(P_base(3, self.block_size))
            self.__left.setP2(4 * P_base(3, self.block_size))
        # Init common properties
        self.__left.setPreFilterCap(self.pre_filter_cap)
        self.__left.setUniquenessRatio(self.uniqueness_ratio)
        self.__left.setDisp12MaxDiff(self.disp_12_max_diff)
        self.__left.setSpeckleRange(self.specke_range)
        self.__left.setSpeckleWindowSize(self.speckle_window_size)
        # Create filter
        self.__filter = createDisparityWLSFilter(self.__left)
        # Create right-oriented matcher
        self.__right = createRightMatcher(self.__left)
        # Set iflter properties
        self.__filter.setLambda(self.wls_lamba)
        self.__filter.setSigmaColor(self.wls_sigma)
        # Get frame crops
        _, _, _, _, _, self.roi1, self.roi2 = cv2.stereoRectify(
            self.m1, self.d1, self.m2, self.d2, self.size, self.r, self.t,
            self.r1, self.r2, self.p1, self.p2, self.q,
            cv2.CALIB_ZERO_DISPARITY, -1, self.size)
        # Get undistortion matrices
        self.m1_ = cv2.initUndistortRectifyMap(self.m1, self.d1, self.r1, self.p1, self.size, cv2.CV_16SC2)
        self.m2_ = cv2.initUndistortRectifyMap(self.m2, self.d2, self.r2, self.p2, self.size, cv2.CV_16SC2)
        # Intersect to get common area
        int_roi = intersection(self.roi1, self.roi2)
        # Convert to bounds
        self.roi = (int_roi[1], int_roi[1] + int_roi[3],
            int_roi[0], int_roi[0] + int_roi[2])


    def compute(self, frames):
        # Undistort left + right frames
        frame_l = cv2.remap(frames[0], self.m1_[0], self.m1_[1], cv2.INTER_LINEAR)
        frame_r = cv2.remap(frames[1], self.m2_[0], self.m2_[1], cv2.INTER_LINEAR)
        # Convert to grayscale
        frame_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY).astype('uint8')
        frame_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY).astype('uint8')
        # Compute left-oriented (base frame is left cam) disparity map
        disp_l = self.__left.compute(frame_l, frame_r)
        # Crop
        disp_l = disp_l[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        # Compute right-oriented (base frame is right cam) disparity map
        disp_r = self.__right.compute(frame_r, frame_l)
        # Crop
        disp_r = disp_r[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        # Use a WLS (Weighted-Least Squares) to find a better depth map
        disp = self.__filter.filter(
            disparity_map_left=disp_l,
            left_view=frame_l,
            disparity_map_right=disp_r
        ).clip(min=0)
        # Scale result, and make 32-bit float (for occ grid)
        disp = (disp * self.factor).astype('float32')
        # Return disparity maps
        return disp, disp_l, disp_r

    def avg_time(self):
        pass
