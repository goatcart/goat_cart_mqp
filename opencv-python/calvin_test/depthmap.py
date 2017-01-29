#!/usr/bin/python3

import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('01L.png',0)
imgR = cv2.imread('01R.png',0)

stereo = cv2.StereoSGBM_create(1, 32, 10)
disparity = stereo.compute(imgL, imgR)

plt.imshow(disparity,'rainbow')
plt.show()
