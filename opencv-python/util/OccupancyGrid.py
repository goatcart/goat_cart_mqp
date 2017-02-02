import numpy as np
import cv2
from .util_fxn import load_mat
from math import floor, sqrt, exp, log

class OccupancyGrid:
    def __init__(self, og_def):
        self.q_mat = load_mat(og_def['Q'])
        self.og_def = og_def
        self.occupancy_size = tuple(og_def['occupancySize'])
        self.x_range = tuple(og_def['xRange'])
        self.y_range = tuple(og_def['yRange'])
        self.z_range = tuple(og_def['zRange'])

    def compute(self, disp):
        image3d = cv2.reprojectImageTo3D(disp, self.q_mat, True)
        occupancy = np.zeros(self.occupancy_size)
        height = np.zeros(self.occupancy_size)

        for i in range(self.occupancy_size[0]):
            for j in range(self.occupancy_size[1]):
                pt = image3d[i,j]
                h = self.og_def['cameraHeight'] - pt[1]
                if pt[0] > self.x_range[1] or pt[0] < self.x_range[0] or \
                    pt[1] > self.y_range[1] or pt[1] < self.y_range[0] or \
                    pt[2] > self.z_range[1] or pt[2] < self.z_range[0]:
                    continue
                scaled_z = (pt[2] - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
                scaled_x = (pt[0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0])

                row = self.occupancy_size[0] - floor(scaled_z * self.occupancy_size[0])
                col = floor(scaled_x * self.occupancy_size[1])
                occupancy[row,col] += 1
                height[row,col] += h

        disp_occ = np.zeros((occupancy.shape[0], occupancy.shape[1]), np.int16)

        x_cam = self.occupancy_size[1] / 2
        y_cam = self.occupancy_size[0] / 2

        c = 0
        for i in range(occupancy.shape[0]):
            for j in range(occupancy.shape[1]):
                if occupancy[i,j] > 0:
                    c += 1
                    x_pt = j
                    y_pt = i
                    dist_to_cam = sqrt((x_cam - y_pt) ** 2 +
                        (y_cam - y_pt) ** 2)
                    adjusted_num = occupancy[i,j] * self.og_def['r'] \
                        / (1 + exp(-dist_to_cam * self.og_def['c']))
                    pij_num = 1 - exp(-(adjusted_num / self.og_def['deltaN']))
                    lij_num = log(pij_num / (1 - pij_num)) if pij_num != 1 else 0

                    avg_height = height[i,j] / occupancy [i,j] if occupancy[i,j] > 0 else 0
                    pij_height = 1 - exp(-avg_height / self.og_def['deltaH'])
                    lij_height = log(pij_height / (1 - pij_height)) if pij_height != 1 else 0

                    avg_prob = self.og_def['wN'] * lij_num + self.og_def['wH'] * lij_height

                    if avg_prob < self.og_def['nt']:
                        disp_occ[i,j] = 0
                    elif lij_num >= self.og_def['lt']:
                        disp_occ[i,j] = 32767
                    else:
                        disp_occ[i,j] = 16383
                else:
                    disp_occ[i,j] = 0
        print(c)
        return disp_occ
