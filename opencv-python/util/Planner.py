# Local imports
from .StereoVision import StereoVision
from .OccupancyGrid import OccupancyGrid
from .VidStream import SourceManager

# External Utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class Planner:
    vid_src = None
    vision = None
    occupancy = None

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
        print("Vision Time = {0}\nOcc Time = {1}\nTotal Time = {2}".format(v_time, o_time, v_time+o_time))

    def render(self):
        # Show results
        fig = plt.figure(1)

        ax_d = fig.add_subplot(121)
        ax_d.imshow(self.vision.disparity)
        ax_d.set_title('Depth Map')

        ax_og = fig.add_subplot(122)
        ax_og.imshow(self.occupancy.occupancy)
        ax_og.set_title('Occupancy Grid')

        plt.show()
