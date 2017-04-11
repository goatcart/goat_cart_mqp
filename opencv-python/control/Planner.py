# Local imports
from .StereoVision import StereoVision
from .OccupancyGrid import OccupancyGrid
from .VidStream import SourceManager
from .CameraCalib import CameraCalib

# External Utils
import matplotlib.pyplot as plt
import numpy as np
import glob
from mpl_toolkits.mplot3d import Axes3D
import time
from enum import Enum
import cv2

DEBUG = False

class Planner:
    vid_src = None
    vision = None
    occupancy = None
    running = False
    calib_count = 10
    frame_titles = ['Left', 'Right']
    title = 'Goat Cart'
    color = np.asarray([63, 63, 63], 'uint8')
    textcolor = (255, 255, 255)
    padding = 150
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 1
    text_thick = 2

    def __init__(self, params):
        # Save params
        self.params = params
        self.src_id = params['matcher']['src']
        # Initialize planning components (start)
        self.vid_src = SourceManager(params['video'])
        self.calib = CameraCalib(
            self.vid_src, self.src_id,
            (8, 6), 10, self.__finish_init)
        self.window = cv2.namedWindow('GoatCart')

    def update(self):
        # Update cameras
        self.vid_src.update()
        # Planning update
        if self.calib.is_ready() and self.vision is not None:
            self.vision.update()
            self.occupancy.update()

    def __finish_init(self):
        self.frame_titles = ['Left', 'Disparity Map', 'Occupancy Grid', 'Right']
        self.title = 'CartVision TM'
        self.vision = StereoVision(self.vid_src, self.calib, self.params['matcher'])
        self.occupancy = OccupancyGrid(self.vision, self.calib, self.params['occupancyGrid'])

    def get_title_size(self, title):
        return cv2.getTextSize(title, self.font, self.text_scale, self.text_thick)[0]

    def combine_frames(self, frames, max_w):
        # Get a copy of each frame so that original is not disturbed
        frames = list(map(lambda x: x.copy(), frames))
        max_w = int(max_w / len(frames))
        max_w_f = max_w - self.padding
        f_sizes = list(map(lambda f: f.shape[:2], frames))
        scales = list(map(lambda sz: min(max_w_f / sz[1], 1), f_sizes))
        for i in range(len(frames)):
            if scales[i] < 1:
                f_sizes[i] = [int(f_sizes[i][0] * scales[i]), int(f_sizes[i][1] * scales[i])]
                frames[i] = cv2.resize(frames[i], (f_sizes[i][1], f_sizes[i][0]))
            if len(frames[i].shape) == 2:
                frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2RGB)
        size = [max(sizes) for sizes in zip(*f_sizes)]
        max_w = min(max_w, size[1] + self.padding)
        max_h = size[0] + self.padding
        frame = np.ones((max_h, max_w * len(frames), 3), np.uint8) * self.color
        for i in range(len(frames)):
            x = int((max_w - f_sizes[i][1]) / 2) + max_w * i
            y = int((max_h - f_sizes[i][0]) / 2)
            h, w = f_sizes[i]
            frame[ y : y + h, x : x + w ] = frames[i]
            title_w, title_h = self.get_title_size(self.frame_titles[i])
            title_loc = (x + int((w - title_w) / 2), max_h - title_h - 5)
            cv2.putText(frame, self.frame_titles[i],
                title_loc,
                self.font, self.text_scale, self.textcolor, self.text_thick,
                cv2.LINE_AA)
        title_w, title_h = self.get_title_size(self.title)
        cv2.putText(frame, self.title,
            (int((max_w * len(frames) - title_w) / 2), 20 + title_h), self.font,
            self.text_scale, self.textcolor, self.text_thick,
            cv2.LINE_AA)
        return frame


    # Update display
    def render(self):
        # Calibration done, show processed output
        if self.calib.is_ready() and self.vision is not None:
            frames = [self.vision.frame_l, self.vision.pretty, self.occupancy.pretty, self.vision.frame_r]
        # Still in calib mode, show raw input
        else:
            frames = self.vid_src.get(self.src_id).frames()
        frame = self.combine_frames(frames, 1800)
        cv2.imshow('GoatCart', frame)
        k = cv2.waitKey(25)
        # Quit
        if k == ord('q'):
            self.running = False
        # Use current frame as calibration keyframe
        elif k == ord('c') and not self.calib.is_ready():
            self.calib.update()
            if not self.calib.is_ready():
                self.title = str(self.calib)

    # Main loop
    def start(self):
        if self.running:
            return
        self.running = True
        while self.running:
            self.update()
            self.render()