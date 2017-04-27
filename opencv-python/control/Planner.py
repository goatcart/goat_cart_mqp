# Local imports
from control.StereoVision import StereoVision
from control.OccupancyGrid import OccupancyGrid
from control.VidStream import SourceManager
from control.CameraCalib import CameraCalib
from control import gui

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
            (7, 7), 10, self.__finish_init)
        self.window = cv2.namedWindow('GoatCart')

    def update(self):
        # Update cameras
        self.vid_src.update()
        # Planning update
        if self.calib.is_ready() and self.vision is not None:
            self.vision.update()
            self.occupancy.update()

    def __finish_init(self):
        # Handle calibration finish
        gui.title = 'CartVision TM'
        self.vision = StereoVision(self.vid_src, self.calib, self.params['matcher'])
        self.occupancy = OccupancyGrid(self.vision, self.calib, self.params['occupancyGrid'])

    # Update display
    def render(self):
        # Calibration done, show processed output
        if self.calib.is_ready() and self.vision is not None:
            frame = gui.combine_frames(1800,
                (self.vision.proc_l, 'Left'),
                (self.vision.pretty, 'Disparity Map'),
                (self.occupancy.pretty, 'Occupancy Grid'),
                (self.vision.proc_r, 'Right')
            )
        # Still in calib mode, show raw input
        else:
            frames = self.vid_src.get(self.src_id).frames()
            frame = gui.combine_frames(1800, (frames[0], 'Left'), (frames[1], 'Right'))
        
        cv2.imshow('GoatCart', frame)
        k = cv2.waitKey(25)
        # Quit
        if k == ord('q'):
            self.running = False
        # Use current frame as calibration keyframe
        elif k == ord('c') and not self.calib.is_ready():
            self.calib.update()
            if not self.calib.is_ready():
                gui.title = str(self.calib)

    # Main loop
    def start(self):
        if self.running:
            return
        self.running = True
        while self.running:
            self.update()
            self.render()