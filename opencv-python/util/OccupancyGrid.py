import numpy as np
import scipy as sp
import cv2
from .util_fxn import load_mat
from math import floor, sqrt, exp, log

'''
Translated from C++ occupancy grid implementation by Guilherme Meira

For more information on the algoritm used, look at his Master's Thesis
https://web.wpi.edu/Pubs/ETD/Available/etd-042616-234937/unrestricted/msthesis-guilhermemeira-finalversion.pdf
'''
class OccupancyGrid:
    def __init__(self, cam_props, og_def):
        # Disparity-to-Depth Matrix
        self.q_mat = load_mat(cam_props['Q'])
        # Occupancy Grid Properties
        self.og_def = og_def
        # Dimensions of the occupancy grid
        self.occupancy_size = tuple(og_def['occupancySize'])
        # 3-D bounds of the space used to form the occupancy grid
        self.x_range = tuple(og_def['xRange'])
        self.y_range = tuple(og_def['yRange'])
        self.z_range = tuple(og_def['zRange'])
        # The y position of the camera
        self.cam_h = og_def['cameraHeight']
        self.r = og_def['r']
        self.c = og_def['c']
        # Probability coefficients
        self.delta_n = og_def['deltaN']
        self.delta_h = og_def['deltaH']
        # Weights
        self.w_n = og_def['wN']
        self.w_h = og_def['wH']
        # Limits for occ status
        self.nt = og_def['nt']
        self.lt = og_def['lt']
        # Robot info
        self.robot_width = og_def['robotWidth']
        self.clearance = og_def['clearance']

    def compute(self, disp):
        # Create point cloud
        image3d = cv2.reprojectImageTo3D(disp, self.q_mat, handleMissingValues=True)
        # Point cloud in box
        pc = []
        occupancy = np.zeros(self.occupancy_size)
        height = np.zeros(self.occupancy_size)

        # For each point in image
        for i in range(image3d.shape[0]):
            for j in range(image3d.shape[1]):
                pt = image3d[i,j]
                # pt.y is the y offset of the point relative the the camera
                # it is also the opposite of the actual value (down is +)
                h = self.cam_h - pt[1]
                # Is the point in the 3d box in front of the camera?
                if pt[0] > self.x_range[1] or pt[0] < self.x_range[0] or \
                    h > self.y_range[1] or h < self.y_range[0] or \
                    pt[2] > self.z_range[1] or pt[2] < self.z_range[0]:
                    continue
                # Scale z to be in the range [0, 1]
                scaled_z = (pt[2] - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
                # Scale x to be in the range [0, 1]
                scaled_x = (pt[0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0])

                # Which row? Scale to fit grid width and flip (z = 0 is at last row)
                row = int(self.occupancy_size[0] - floor(scaled_z * self.occupancy_size[0]))
                # Which column? Scale to fit grid width
                col = int(floor(scaled_x * self.occupancy_size[1]))
                # One more point in that cell
                occupancy[row,col] += 1
                # Sum of heights in that cell (to find average)
                height[row,col] += h
                pc.append((pt[0], h, pt[2]))

        # Total # of valid points
        print(len(pc))
        # Final Occupancy Grid
        disp_occ = np.zeros(occupancy.shape, np.int16)

        # Location of camera on occupancy grid
        x_cam = self.occupancy_size[0] / 2
        y_cam = self.occupancy_size[1]
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

        # We don't care about the 'unknowns'
        disp_occ[disp_occ == 1] = 0

        # Get rid of small clusters by eroding, then dilating
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        disp_occ = cv2.morphologyEx(disp_occ, cv2.MORPH_OPEN, kernel)

        # Calculate width of cell relative to original 3d box width
        cell_width = (self.x_range[1] - self.x_range[0]) / occupancy.shape[1]
        # Calculate dilation factor ('radius' of robot + bit extra)
        # Then convert from width in terms of box to width in terms of cells
        dilation_n = int((self.robot_width / 2 + self.clearance) / cell_width)

        # Dilate to ensure clearance
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_n + 1, 1))
        disp_occ = cv2.dilate(disp_occ, dilation_kernel)

        # Return the occupancy grid and the point cloud
        return disp_occ, pc
