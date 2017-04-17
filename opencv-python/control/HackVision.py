import numpy as np
import cv2

def detect_cone(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

     # define range of blue color in HSV
    lower_blue = np.array([0,100,127])
    upper_blue = np.array([40,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    return res, (float(np.count_nonzero(mask)) / mask.size)