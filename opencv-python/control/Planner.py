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

    def __init__(self, params):
        # Save params
        self.params = params
        self.src_id = params['matcher']['src']
        self.__init_display()
        # Initialize planning components
        self.vid_src = SourceManager(params['video'], params['src_props'])
        self.calib = CameraCalib(
            self.vid_src, self.src_id,
            (6,7), 10, self.__finish_init)

    def update(self):
        # Update planning components
        self.vid_src.update()
        if self.calib.is_ready():
            self.vision.update()
            self.occupancy.update()
    
    def __init_display(self):
        # Show results
        self.fig = plt.figure(1)
        self.fig.canvas.mpl_connect('key_press_event', self.__kp)

        self.ax_l = self.fig.add_subplot(121)
        self.ax_l.set_title('Left')

        self.ax_r = self.fig.add_subplot(122)
        self.ax_r.set_title('Right')

        plt.ion()
        plt.show()

    def __finish_init(self):
        self.vision = StereoVision(self.vid_src, self.calib, params['matcher'])
        self.occupancy = OccupancyGrid(self.vision, self.calib, params['occupancyGrid'])

    def __kp(self, evt):
        if evt.key == 'q':
            print ('close')
            self.running = False
        if evt.key == 'c' and not self.calib.is_ready():
            self.calib.update()

    def render(self):
        if self.calib.is_ready():
            self.ax_l.imshow(self.vision.disparity)
            self.ax_r.imshow(self.occupancy.occupancy)
        else:
            frame_l, frame_r = self.vid_src.get(self.src_id).frames()
            self.ax_l.imshow(cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB))
            self.ax_r.imshow(cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB))
        plt.pause(0.05)

    def start(self):
        if self.running:
            return
        self.running = True
        self.ax_l.set_title('Disparity Map')
        self.ax_r.set_title('Occupancy Grid')
        while self.running:
            self.update()
            self.render()