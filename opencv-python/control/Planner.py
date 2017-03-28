# Local imports
from .StereoVision import StereoVision
from .OccupancyGrid import OccupancyGrid
from .VidStream import SourceManager

# External Utils
import matplotlib.pyplot as plt
import numpy as np
import glob
from mpl_toolkits.mplot3d import Axes3D
import time
from enum import Enum

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
        self.__init_display()
        # Initialize planning components
        self.vid_src = SourceManager(params['video'], params['src_props'])
        self.calib = CameraCalib(self.vid_src, params['matcher']['src'])
        self.vision = StereoVision(self.vid_src, self.calib, params['matcher'])
        self.occupancy = OccupancyGrid(self.vision, self.calib, params['occupancyGrid'])

    def update(self):
        # Update planning components
        self.vid_src.update()
        if self.calib.is_done():
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

    def __kp(self, evt):
        if evt.key == 'q':
            print ('close')
            self.running = False
        if evt.key == 's' and not self.calib.is_done():
            self.calib.update()

    def render(self):
        self.ax_l.imshow(self.vision.disparity)
        self.ax_r.imshow(self.occupancy.occupancy)
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