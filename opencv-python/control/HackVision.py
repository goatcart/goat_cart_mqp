import numpy as np
import cv2

blue = [255, 0, 0]
green = [0, 255, 0]
red = [0, 0, 255]
orange_c = 0
ksize = 15
thresh = 0.038
factor = 0.05

def detect_cone(frame):
    global orange_c

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

     # define range of blue color in HSV
    lower_blue = np.array([0,100,127])
    upper_blue = np.array([40,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    orange_c = orange_c * (1 - factor) + (float(np.count_nonzero(mask)) / mask.size) * factor
    color = np.ones(frame.shape)
    if orange_c > thresh:
        color *= green
    else:
        color *= blue
    color = color.astype('uint8')
    mask_inv = cv2.bitwise_not(mask)

    # Bitwise-AND mask and original image
    res = cv2.add(cv2.bitwise_and(frame, frame, mask=mask), cv2.bitwise_and(color, color, mask=mask_inv))

    return res, orange_c, orange_c > thresh