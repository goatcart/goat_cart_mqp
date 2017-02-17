#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util.VidStream import SourceManager
from util.StereoVision import StereoVision
from util.OccupancyGrid import OccupancyGrid
import json

stream = open('params.json')
fs = json.load(stream)
ocs = fs['occupancyGrid']

vs = SourceManager(fs['video'])
sv = StereoVision(vs)
og = OccupancyGrid(fs['occupancyGrid'])

main_src = fs['matcher']['src']

f = vs.get_frame(main_src)
d, dl, dr = sv.compute(f)

occ, im3d = og.compute(d)
r3d = im3d.ravel()

r3d = [
    r3d[0::3],
    r3d[1::3],
    r3d[2::3]
]

fig = plt.figure(1)

ax_c = fig.add_subplot(221)
ax_c.imshow(dl, cmap='gray')
ax_c.set_title('Depth Map')

ax_d = fig.add_subplot(222)
ax_d.imshow(d, cmap='gray')
ax_d.set_title('Depth Map (final)')

ax_3d = fig.add_subplot(223, projection='3d')
ax_3d.scatter(r3d[0][::50], r3d[1][::50], r3d[2][::50], s=0.25)
ax_3d.set_xlabel('X')
ax_3d.set_xlim(ocs['xRange'])
ax_3d.set_ylim(ocs['yRange'])
ax_3d.set_zlabel('Z')
ax_3d.set_zlim(ocs['zRange'])
ax_3d.set_title('Point Cloud (Occ)')
ax_3d.view_init(5, -85)

ax_og = fig.add_subplot(224)
ax_og.imshow(occ)
ax_og.set_title('Occupancy Grid')

plt.show()
