# Local imports
from control.VidStream import SourceManager
from control.HackVision import detect_cone
from control import gui

# External Utils
import numpy as np
import cv2

DEBUG = False

class Planner:
    vid_src = None
    running = False
    orange_c = 0
    factor = 0.02

    def __init__(self, params):
        # Save params
        self.params = params
        self.src_id = params['matcher']['src']
        # Initialize planning components (start)
        self.vid_src = SourceManager(params['video'])
        self.source = self.vid_src.get(self.src_id)
        gui.frame_titles = ['In', 'Proc']
        self.window = cv2.namedWindow('GoatCart')

    def update(self):
        # Update cameras
        self.vid_src.update()

    # Update display
    def render(self):
        left = self.source.frames()[0]
        proc, cnt = detect_cone(left)
        self.orange_c = self.orange_c * (1 - self.factor) + cnt * self.factor

        gui.frame_titles[1] = 'Proc: {0:.1f}%'.format(self.orange_c*100)
        frames = [left, proc]
        frame = gui.combine_frames(frames, 1800)
        cv2.imshow('GoatCart', frame)
        k = cv2.waitKey(25)
        # Quit
        if k == ord('q'):
            self.running = False
        # Use current frame as calibration keyframe
        elif k == ord('c') and self.proc and not self.calib.is_ready():
            self.calib.update()
            if not self.calib.is_ready():
                self.title = str(self.calib)

    # Main loop
    def start(self):
        if self.running:
            return
        self.running = True
        while self.running:
            self.update()
            self.render()