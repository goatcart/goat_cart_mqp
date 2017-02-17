import numpy as np
import scipy as sp
import cv2
from .util_fxn import load_mat
from math import floor, sqrt, exp, log

class OccupancyGrid:
    def __init__(self, cam_props, og_def):
        self.q_mat = load_mat(cam_props['Q'])
        self.og_def = og_def
        self.occupancy_size = tuple(og_def['occupancySize'])
        self.x_range = tuple(og_def['xRange'])
        self.y_range = tuple(og_def['yRange'])
        self.z_range = tuple(og_def['zRange'])
        self.cam_h = og_def['cameraHeight']
        self.r = og_def['r']
        self.c = og_def['c']
        self.delta_n = og_def['deltaN']
        self.delta_h = og_def['deltaH']
        self.w_n = og_def['wN']
        self.w_h = og_def['wH']
        self.nt = og_def['nt']
        self.lt = og_def['lt']
        self.robot_width = og_def['robotWidth']
        self.clearance = og_def['clearance']

    def compute(self, disp):
        image3d = cv2.reprojectImageTo3D(disp, self.q_mat, handleMissingValues=True)
        occupancy = np.zeros(self.occupancy_size)
        height = np.zeros(self.occupancy_size)

        c = 0
        for i in range(image3d.shape[0]):
            for j in range(image3d.shape[1]):
                pt = image3d[i,j]
                h = self.cam_h - pt[1]
                if pt[0] > self.x_range[1] or pt[0] < self.x_range[0] or \
                    h > self.y_range[1] or h < self.y_range[0] or \
                    pt[2] > self.z_range[1] or pt[2] < self.z_range[0]:
                    continue
                scaled_z = (pt[2] - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
                scaled_x = (pt[0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0])

                row = int(self.occupancy_size[0] - floor(scaled_z * self.occupancy_size[0]))
                col = int(floor(scaled_x * self.occupancy_size[1]))
                occupancy[row,col] += 1
                height[row,col] += h
                c += 1

        print('Valid Points:', c)
        disp_occ = np.zeros(occupancy.shape, np.int16)

        x_cam = self.occupancy_size[0] / 2
        y_cam = self.occupancy_size[1] / 2
        for i in range(occupancy.shape[0]):
            for j in range(occupancy.shape[1]):
                if occupancy[i,j] > 0:
                    x_pt = j
                    y_pt = i
                    dist_to_cam = sqrt((x_cam - x_pt) ** 2 +
                        (y_cam - y_pt) ** 2)
                    adjusted_num = occupancy[i,j] * self.r / (1 + exp(-dist_to_cam * self.c))
                    pij_num = 1 - exp(-(adjusted_num / self.delta_n))
                    lij_num = log(pij_num / (1 - pij_num)) if pij_num != 1 else 0

                    avg_height = height[i,j] / occupancy[i,j] if occupancy[i,j] > 0 else 0
                    pij_height = 1 - exp(-avg_height / self.delta_h)
                    lij_height = log(pij_height / (1 - pij_height)) if pij_height != 1 else 0

                    avg_prob = self.w_n * lij_num + self.w_h * lij_height

                    if avg_prob < self.nt: #free
                        disp_occ[i,j] = 0
                    elif lij_num >= self.lt: #occupied
                        disp_occ[i,j] = 2
                    else: #unknown
                        disp_occ[i,j] = 1
                else: #free
                    disp_occ[i,j] = 0

        disp_occ[disp_occ == 1] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        disp_occ = cv2.morphologyEx(disp_occ, cv2.MORPH_OPEN, kernel)

        cell_width = (self.x_range[1] - self.x_range[0]) / occupancy.shape[1]
        dilation_n = int((self.robot_width / 2 + self.clearance) / cell_width)

        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_n + 1, 1))
        disp_occ = cv2.dilate(disp_occ, dilation_kernel)

        return disp_occ, image3d
