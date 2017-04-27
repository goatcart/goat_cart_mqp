import numpy as np
import numpy.ma as ma
import scipy as sp
import cv2
from control.util_fxn import load_mat, div0
from math import floor, sqrt, exp, log
import time

'''
Translated from C++ occupancy grid implementation by Guilherme Meira

For more information on the algoritm used, look at his Master's Thesis
https://web.wpi.edu/Pubs/ETD/Available/etd-042616-234937/unrestricted/msthesis-guilhermemeira-finalversion.pdf
'''
class OccupancyGrid:
    def __init__(self, vision, calib, occupancy_params):
        self.__vision = vision
        self.__calib = calib
        self.__cfg = occupancy_params['cfg']
        self.__c1 = occupancy_params['clean_up']
        self.__c2 = occupancy_params['dilate']
        occupancy_params = occupancy_params['o_props'][self.__cfg]
        self.__params = occupancy_params
        # Disparity-to-Depth Matrix
        self.q_mat = self.__calib.q
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
        # Init coords
        self.coords = np.zeros((200, 200, 2), np.int16)
        for i in range(200):
            self.coords[i, :, 0] = i
            self.coords[:, i, 1] = i
        # Location of camera on occupancy grid
        x_cam = self.occupancy_size[0] / 2
        y_cam = self.occupancy_size[1]
        # Distance from each point to camera
        self.dist_to_cam = np.sqrt(np.sum(np.square(self.coords - [y_cam, x_cam])))

    def update(self):
        disparity = self.__vision.disparity
        # Create point cloud
        image3d = cv2.reprojectImageTo3D( \
            disparity, self.q_mat, handleMissingValues=True)
        
        ### Generate matrices
        
        # Create matrices
        occupancy  = np.zeros(self.occupancy_size) # Number of points in cell
        height     = np.zeros(occupancy.shape) # Total height of elements in cell
        disp_occ   = np.zeros(occupancy.shape, np.uint8) # Final Occupancy Grid
        lij_num    = np.zeros(occupancy.shape) # Log-odds that cell is occupied (elements)
        avg_height = np.zeros(occupancy.shape) # Average height of cell
        lij_height = np.zeros(occupancy.shape) # Log-odds that cell is occupied (height)

        # Filter PC coords so that (x, z) is in the range [0, 1]
        # Y is adusted because input is height below camera
        s_pts = np.dstack((
            (image3d[:, :, 0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]),
            self.cam_h - image3d[:, :, 1],
            (image3d[:, :, 2] - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
        ))

        # Determine which points are valid (inside box)
        invalid = np.any(np.dstack((
            s_pts[:, :, 0] < 0, s_pts[:, :, 0] > 1,
            s_pts[:, :, 2] < 0, s_pts[:, :, 2] > 1,
            s_pts[:, :, 1] < self.y_range[0], s_pts[:, :, 1] > self.y_range[1]
        )), axis=2)

        # Convert (z, x) into (row, col) for occ grid
        scaledCoords = np.dstack((
            self.occupancy_size[0] - np.floor(s_pts[:, :, 2] * self.occupancy_size[0]),
            np.floor(s_pts[:, :, 0] * self.occupancy_size[1])
        )).astype(int)

        # Sum up heights and # of elements in each cell
        for pt, h in zip(scaledCoords[~invalid], s_pts[:, :, 1][~invalid]):
            height[pt[0], pt[1]] += h
            occupancy[pt[0], pt[1]] += 1

        ### Generate occupancy grid

        # Adjust # of points in cell using sigmoid function
        adjusted_num = occupancy * self.r / (1 + np.exp(-self.dist_to_cam * self.c))

        # delta N is a parameter for the probability calculation
        pij_num = 1 - np.exp(-(adjusted_num / self.delta_n))
        
        # Convert probability to 'log-odds' (logit)
        with np.errstate(divide='ignore', invalid='ignore'):
            lij_num = pij_num / (1 - pij_num)
            lij_num[~np.isfinite(lij_num)] = 1
            lij_num = np.log(lij_num)

        # Get average height in cell
        avg_height = div0(height, occupancy)
        
        # Calculate probability of cell being occupied,
        # based on average height of cell occupants

        # delta H is a parameter for the probability calculation
        pij_height = 1 - np.exp(-avg_height / self.delta_h)
        
        # Convert probability to 'log-odds' (logit)
        with np.errstate(divide='ignore', invalid='ignore'):
            lij_height = pij_height / (1 - pij_height)
            lij_height[~np.isfinite(lij_height)] = 1
            lij_height = np.log(lij_height)

        # Estimate the probability that something is in the cell
        # This is a weighted average, so w_n + w_h = 1
        avg_prob = self.w_n * lij_num + self.w_h * lij_height

        # disp_occ[occupancy > 0] = 127 # Unknown
        disp_occ[lij_num >= self.lt] = 255 # Cell is occupied (probably)
        disp_occ[avg_prob < self.nt] = 0 # Nothing here (probably)

        # We don't care about the 'unknowns'
        # disp_occ[disp_occ == 127] = 0

        if self.__c1:
            # Get rid of small clusters by eroding, then dilating
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            disp_occ = cv2.morphologyEx(disp_occ, cv2.MORPH_OPEN, kernel)

            if self.__c2:
                # Calculate width of cell relative to original 3d box width
                cell_width = (self.x_range[1] - self.x_range[0]) / occupancy.shape[1]
                # Calculate dilation factor ('radius' of robot + bit extra)
                # Then convert from width in terms of box to width in terms of cells
                dilation_n = int((self.robot_width / 2 + self.clearance) / cell_width)

                # Dilate to ensure clearance
                dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_n + 1, 1))
                disp_occ = cv2.dilate(disp_occ, dilation_kernel)

        # Return the occupancy grid and the point cloud
        self.occupancy = disp_occ
        self.pretty = cv2.resize(disp_occ, (self.occupancy_size[1] * 2, self.occupancy_size[0] * 2))
