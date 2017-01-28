#!/usr/bin/env python3

import numpy as np
import cv2
from matplotlib import pyplot as plt
from util.VidStream import VidStream
from util.StereoVision import StereoVision
import json

stream = open('params.json')
fs = json.load(stream)
vs = VidStream(fs['video'])
sv = StereoVision()

f = vs.get_frame()
d, dl, dr = sv.compute(f)

plt.figure(1)

plt.subplot(221)
plt.imshow(f[0])

plt.subplot(222)
plt.imshow(f[1])

plt.subplot(223)
plt.imshow(d)

plt.subplot(224)
plt.imshow(dl)

plt.show()
