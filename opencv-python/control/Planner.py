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
        proc, orange_c, res = detect_cone(left)
        text = ''
        color = [255, 255, 255]
        if res:
            text = ' [ STOP ]'
            color = [0, 128, 255]
        title = 'Orange: {:.1f}%{}'.format(orange_c*100, text)
        frames = [left, proc]
        frame = gui.combine_frames(1800, (left, 'Capture'), (proc, title, color))
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