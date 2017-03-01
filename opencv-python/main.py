#!/usr/bin/env python3

import numpy as np
from util.Planner import Planner
import json

stream = open('params.json')
params = json.load(stream)

planner = Planner(params)

planner.update()
planner.render()

'''
i_occ = fs['occupancyGrid']
i_matcher = fs['matcher']
cam_props = fs['src_props'][i_matcher['prop']]

vid_stream = SourceManager(fs['video'])
vision = StereoVision(vid_stream)
occ_grid = OccupancyGrid(cam_props, i_occ)

main_src = fs['matcher']['src']

f = vid_stream.get_frame(main_src)
d, dl, dr = vision.compute(f)

occ, im3d = occ_grid.compute(d)
r3d = [
    [pt[0] for pt in im3d],
    [pt[1] for pt in im3d],
    [pt[2] for pt in im3d]
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
ax_3d.set_xlim(i_occ['xRange'])
ax_3d.set_ylim(i_occ['yRange'])
ax_3d.set_zlabel('Z')
ax_3d.set_zlim(i_occ['zRange'])
ax_3d.set_title('Point Cloud (Occ)')
ax_3d.view_init(5, -85)

ax_og = fig.add_subplot(224)
ax_og.imshow(occ)
ax_og.set_title('Occupancy Grid')

plt.show()
'''
