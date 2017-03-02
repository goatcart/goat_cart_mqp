import numpy as np
import numpy.ma as ma
import scipy as sp
import cv2
from .util_fxn import load_mat
from math import floor, sqrt, exp, log
import time

'''
Translated from C++ occupancy grid implementation by Guilherme Meira

For more information on the algoritm used, look at his Master's Thesis
https://web.wpi.edu/Pubs/ETD/Available/etd-042616-234937/unrestricted/msthesis-guilhermemeira-finalversion.pdf
'''
class OccupancyGrid:
    def __init__(self, vision, occupancy_params):
        self.__vision = vision
        self.__cfg = occupancy_params['cfg']
        occupancy_params = occupancy_params['o_props'][self.__cfg]
        self.__params = occupancy_params
        # Disparity-to-Depth Matrix
        self.q_mat = vision.q
        # Dimensions of the occupancy grid
        self.occupancy_size = tuple(occupancy_params['occupancySize'])
        # 3-D bounds of the space used to form the occupancy grid
        self.x_range = tuple(occupancy_params['xRange'])
        self.y_range = tuple(occupancy_params['yRange'])
        self.z_range = tuple(occupancy_params['zRange'])
        # The y position of the camera
        self.cam_h = occupancy_params['cameraHeight']
        self.r = occupancy_params['r']
        self.c = occupancy_params['c']
        # Probability coefficients
        self.delta_n = occupancy_params['deltaN']
        self.delta_h = occupancy_params['deltaH']
        # Weights
        self.w_n = occupancy_params['wN']
        self.w_h = occupancy_params['wH']
        # Limits for occ status
        self.nt = occupancy_params['nt']
        self.lt = occupancy_params['lt']
        # Robot info
        self.robot_width = occupancy_params['robotWidth']
        self.clearance = occupancy_params['clearance']

    def update(self):
        disparity = self.__vision.disparity
        # Create point cloud
        image3d = cv2.reprojectImageTo3D( \
            disparity, self.q_mat, handleMissingValues=True)
        occupancy = np.zeros(self.occupancy_size)
        height = np.zeros(self.occupancy_size)
        # Final Occupancy Grid
        disp_occ = np.zeros(occupancy.shape, np.int16)

        start = time.clock()
        # pt.y is the y offset of the point relative the the camera
        # it is also the opposite of the actual value (down is +)
        # Scale Z+X to [0,1]

        s_pts = np.dstack((
            (image3d[:,:,0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]),
            self.cam_h - image3d[:,:,1],
            (image3d[:,:,2] - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
        ))

        invalid = np.any(np.dstack((
            s_pts[:,:,0] < 0, s_pts[:,:,0] > 1,
            s_pts[:,:,2] < 0, s_pts[:,:,2] > 1,
            s_pts[:,:,1] < self.y_range[0], s_pts[:,:,1] > self.y_range[1]
        )), axis=2)

        scaledCoords = np.dstack((
            self.occupancy_size[0] - np.floor(s_pts[:,:,2] * self.occupancy_size[0]),
            np.floor(s_pts[:,:,0] * self.occupancy_size[1])
        )).astype(int)
        span = time.clock() - start
        print('Mat Time = {0}'.format(span))

        start = time.clock()
        for pt, h in zip(scaledCoords[~invalid], s_pts[:,:,1][~invalid]):
            height[pt[0],pt[1]] += h
            occupancy[pt[0],pt[1]] += 1
        span = time.clock() - start
        print('Loop1 Time = {0}'.format(span))

        # Location of camera on occupancy grid
        x_cam = self.occupancy_size[0] / 2
        y_cam = self.occupancy_size[1]

        start = time.clock()
        # For each cell in grid
        for i in range(occupancy.shape[0]):
            for j in range(occupancy.shape[1]):
                # Is there stuff at this cell?
                if occupancy[i,j] > 0:
                    # Get local copy of coords
                    x_pt = j
                    y_pt = i
                    # Euler's formula
                    dist_to_cam = sqrt((x_cam - x_pt) ** 2 +
                        (y_cam - y_pt) ** 2)
                    # Adjust the number of points in cell using sigmoid fxn
                    # r and c are control coefficients
                    adjusted_num = occupancy[i,j] * self.r / (1 + exp(-dist_to_cam * self.c))
                    # Calculate probability of cell being occupied,
                    # based on adjusted number of cell occupants

                    # delta N is a parameter for the probability calculation
                    pij_num = 1 - exp(-(adjusted_num / self.delta_n))
                    # Convert probability to 'log-odds' (logit)
                    lij_num = log(pij_num / (1 - pij_num)) if pij_num != 1 else 0

                    # Get average height in cell
                    avg_height = height[i,j] / occupancy[i,j] if occupancy[i,j] > 0 else 0
                    # Calculate probability of cell being occupied,
                    # based on average height of cell occupants

                    # delta H is a parameter for the probability calculation
                    pij_height = 1 - exp(-avg_height / self.delta_h)
                    # Convert probability to 'log-odds' (logit)
                    lij_height = log(pij_height / (1 - pij_height)) if pij_height != 1 else 0

                    # Estimate the probability that something is in the cell
                    # This is a weighted average, so w_n + w_h = 1
                    avg_prob = self.w_n * lij_num + self.w_h * lij_height

                    if avg_prob < self.nt: # Nothing here (probably)
                        disp_occ[i,j] = 0
                    elif lij_num >= self.lt: # Cell is occupied (probably)
                        disp_occ[i,j] = 2
                    else: # We don't know exactly
                        disp_occ[i,j] = 1
                else: # There is definitely nothing here
                    disp_occ[i,j] = 0
        span = time.clock() - start
        print('Loop2 Time = {0}'.format(span))

        # We don't care about the 'unknowns'
        disp_occ[disp_occ == 1] = 0

        # Get rid of small clusters by eroding, then dilating
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        disp_occ = cv2.morphologyEx(disp_occ, cv2.MORPH_OPEN, kernel)

        # Calculate width of cell relative to original 3d box width
        cell_width = (self.x_range[1] - self.x_range[0]) / occupancy.shape[1]
        # Calculate dilation factor ('radius' of robot + bit extra)
        # Then convert from width in terms of box to width in terms of cells
        dilation_n = int((self.robot_width / 2 + self.clearance) / cell_width)

        # Dilate to ensure clearance
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_n + 1, 1))
        #disp_occ = cv2.dilate(disp_occ, dilation_kernel)

        # Return the occupancy grid and the point cloud
        self.occupancy = disp_occ
