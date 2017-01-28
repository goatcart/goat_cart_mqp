import cv2
import numpy as np

class SourceBase:
    def __init__(self, i_src):
        self.__info = i_src

    def name(self):
        if 'name' in self.__info:
            return self.__info['name']
        return 'Unkown'

class DummySource(SourceBase):
    def __init__(self, i_src):
        super().__init__(i_src)
        self.__img = cv2.imread(i_src['path'])

    def f_grab(self):
        pass

    def f_retrieve(self):
        return np.copy(self.__img)

class VidSource(SourceBase):
    def __init__(self, i_src, i_vid):
        super().__init__(i_src)
        self.__src = cv2.VideoCapture(i_src['id'])
        self.__src.set(cv2.CAP_PROP_FRAME_WIDTH,  i_vid['src-rez'][0])
        self.__src.set(cv2.CAP_PROP_FRAME_HEIGHT, i_vid['src-rez'][1])

    def f_grab(self):
        self.__src.grab()

    def f_retrieve(self):
        ret, frame = self.__src.retrieve()
        return frame

class VidStream:
    __src = []
    def __init__(self, vid_info):
        self.__info = vid_info
        for i_src in vid_info['src']:
            if i_src['type'] == 'dummy':
                self.__src.append(DummySource(i_src))
            else:
                self.__src.append(VidSource(i_src, vid_info))

    def get_frame(self):
        for src in self.__src:
            src.f_grab()
        return [src.f_retrieve() for src in self.__src]
