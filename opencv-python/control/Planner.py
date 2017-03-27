# Local imports
from .StereoVision import StereoVision
from .OccupancyGrid import OccupancyGrid
from .VidStream import SourceManager

# External Utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

DEBUG = False

class Planner:
    vid_src = None
    vision = None
    occupancy = None
    running = False

    def __init__(self, params):
        # Save params
        self.params = params
        # Initialize planning components
        self.vid_src = SourceManager(params['video'], params['src_props'])
        self.vision = StereoVision(self.vid_src, params['matcher'])
        self.occupancy = OccupancyGrid(self.vision, params['occupancyGrid'])

    def update(self):
        # Update planning components (and time)
        self.vid_src.update()
        v_start = time.clock()
        self.vision.update()
        v_time = time.clock() - v_start
        o_start = time.clock()
        self.occupancy.update()
        o_time = time.clock() - o_start
        if DEBUG:
            print("Vision Time = {0}\nOcc Time = {1}\nTotal Time = {2}".format(v_time, o_time, v_time+o_time))
    
    def __init_display(self):
        # Show results
        self.fig = plt.figure(1)
        self.fig.canvas.mpl_connect('key_press_event', self.__kp)

        self.ax_d = self.fig.add_subplot(121)
        self.ax_d.set_title('Depth Map')

        self.ax_og = self.fig.add_subplot(122)
        self.ax_og.set_title('Occupancy Grid')

        plt.ion()
        plt.show()

    def __kp(self, evt):
        if evt.key == 'q':
            print ('close')
            self.running = False
        if evt.key == 's':
            pass

    def render(self):
        self.ax_d.imshow(self.vision.disparity)
        self.ax_og.imshow(self.occupancy.occupancy)
        plt.pause(0.05)

    def start(self):
        if self.running:
            return
        self.running = True
        self.__init_display()
        while self.running:
            self.update()
            self.render()