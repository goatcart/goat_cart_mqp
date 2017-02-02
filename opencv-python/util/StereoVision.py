import cv2
from cv2.ximgproc import createDisparityWLSFilter, createRightMatcher
import numpy as np
from enum import Enum
import json

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
    factor = 0

    def __init__(self, src):
        self.__src = src
        self.__load_settings()
        self.__init_matcher()

    def __load_settings(self):

        fs = json.load(open('params.json', 'r'))
        i_matcher = fs['matcher']

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

        default_width = self.__src.get(self.src_id).size()[0]
        self.num_disp            = calc_disp(default_width, self.scale_factor)
        self.pre_filter_cap      = i_matcher['preFilterCap']
        self.uniqueness_ratio    = i_matcher['uniquenessRatio']
        self.disp_12_max_diff    = i_matcher['disp12MaxDiff']
        self.specke_range        = i_matcher['speckleRange']
        self.speckle_window_size = i_matcher['speckleWindowSize']

    def __init_matcher(self):
        if self.mode == MatcherType.stereo_bm:
            self.__left = cv2.StereoBM_create(self.num_disp, self.block_size)
        else:
            self.__left = cv2.StereoSGBM_create(0, self.num_disp, self.block_size)
            self.__left.setP1(P_base(1, self.block_size))
            self.__left.setP2(4 * P_base(1, self.block_size))
        self.__left.setPreFilterCap(self.pre_filter_cap)
        self.__left.setUniquenessRatio(self.uniqueness_ratio)
        self.__left.setDisp12MaxDiff(self.disp_12_max_diff)
        self.__left.setSpeckleRange(self.specke_range)
        self.__left.setSpeckleWindowSize(self.speckle_window_size)
        self.__filter = createDisparityWLSFilter(self.__left)
        self.__right = createRightMatcher(self.__left)
        self.__filter.setLambda(8000)
        self.__filter.setSigmaColor(1.0)


    def compute(self, frames):
        frame_l = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        frame_r = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
        disp_l = self.__left.compute(frame_l, frame_r)
        disp_r = self.__right.compute(frame_r, frame_l)
        disp = self.__filter.filter(
            disparity_map_left=disp_l,
            left_view=frame_l,
            disparity_map_right=disp_r
        ).clip(min=0)
        min_v, max_v, _, _ = cv2.minMaxLoc(disp)
        self.factor = 63 / (max_v - min_v) + self.factor * 0.75
        disp = (disp * self.factor).astype('uint8')
        return disp, disp_l, disp_r

    def avg_time(self):
        pass
