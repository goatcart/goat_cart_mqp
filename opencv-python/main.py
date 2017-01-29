#!/usr/bin/env python3

import numpy as np
import cv2
from matplotlib import pyplot as plt
from util.VidStream import SourceManager
from util.StereoVision import StereoVision
import json

stream = open('params.json')
fs = json.load(stream)
vs = SourceManager(fs['video'])
sv = StereoVision(vs)

main_src = fs['matcher']['src']

f = vs.get_frame(main_src)
d, dl, dr = sv.compute(f)

plt.figure(1)

plt.subplot(221)
plt.imshow(f[0])
plt.title('Left Cam')

plt.subplot(222)
plt.imshow(d)
plt.title('Depth Map (final)')

plt.subplot(223)
plt.imshow(dl)
plt.title('Depth Map (left)')

plt.subplot(224)
plt.imshow(dr)
plt.title('Depth Map (right)')

plt.show()
