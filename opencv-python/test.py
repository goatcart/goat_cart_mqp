#!/bin/python3

import cv2
from cv2.ximgproc import createDisparityWLSFilterGeneric, createRightMatcher
import numpy as np
import matplotlib.pyplot as plt

num_disp = 80
block_size = 5
disp_12_max_diff = 100
speckle_range = 10
speckle_window_size = 1000
p1 = 200
p2 = 800
pre_filter_cap = 1
uniqueness_ratio = 19

matcher = cv2.StereoSGBM_create(0, num_disp, block_size)
matcher.setDisp12MaxDiff(disp_12_max_diff)
matcher.setSpeckleRange(speckle_range)
matcher.setSpeckleWindowSize(speckle_window_size)
matcher.setP1(p1)
matcher.setP2(p2)
matcher.setPreFilterCap(pre_filter_cap)
matcher.setUniquenessRatio(uniqueness_ratio)


filt = createDisparityWLSFilterGeneric(True)
filt.setLambda(8000)
filt.setSigmaColor(0.8)

m2 = createRightMatcher(matcher)

left = cv2.imread('../data/test_images/left000.png', cv2.IMREAD_COLOR)
right = cv2.imread('../data/test_images/right000.png', cv2.IMREAD_COLOR)
left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype('uint8')
right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY).astype('uint8')

disp = matcher.compute(left, right)
disp2 = m2.compute(right, left)

disp3 = filt.filter(
    disparity_map_left=disp, left_view=left,
    disparity_map_right=disp2, right_view=right
)

fig = plt.figure(1)

ax1 = fig.add_subplot(121)
ax1.imshow(disp)


ax2 = fig.add_subplot(122)
ax2.imshow(disp3)

plt.show()
