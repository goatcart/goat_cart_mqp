import cv2
import numpy as np

__sides = ['left', 'right']

class SourceBase:
    def __init__(self, i_src):
        self.__info = i_src

    def name(self):
        if self.__info and 'name' in self.__info:
            return self.__info['name']
        return 'Unkown'

    def scale(self):
        if self.__info and 'scale' in self.__info:
            return self.__info['scale']
        return 1.0

    def size(self):
        if self.__info and 'rez' in self.__info:
            return self.__info['rez']
        return (480, 360)

class DummySource(SourceBase):
    def __init__(self, i_src):
        super().__init__(i_src)

    def f_grab(self):
        pass

    def f_retrieve(self):
        return None

class ImageSource(SourceBase):
    def __init__(self, i_src):
        super().__init__(i_src)
        path = i_src['path']
        self.__img = [cv2.imread(path + src, 1) for src in i_src['src']]
        self.__shape = self.__img[0].shape

    def size(self):
        return (self.__shape[1], self.__shape[0])

    def f_grab(self):
        pass

    def f_retrieve(self):
        return [np.copy(img) for img in self.__img]

### Yeah, needs to be improved
class VidSource(SourceBase):
    __src = []
    def __init__(self, i_src):
        super().__init__(i_src)
        for i in i_src['src']:
            cam = cv2.VideoCapture(i_src['ident'][0])
            cam.set(cv2.CAP_PROP_FRAME_WIDTH,  i_src['src-rez'][0])
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, i_src['src-rez'][1])
            self.__src.add(cam)

    def f_grab(self):
        for src in self.__src:
            src.grab()

    def f_retrieve(self):
        success = True
        frame = []
        for src in self.__src:
            ret, frame_ = self.__src.retrieve()
            frame.add(frame_)
        return frame


class SourceManager:
    __streams = {}
    def __init__(self, vid_info):
        self.__info = vid_info
        for i_src in vid_info:
            if i_src['enabled']:
                src = DummySource(None)
                if i_src['type'] == 'image':
                    src = ImageSource(i_src)
                elif i_src['type'] == 'cap':
                    src = VidSource(i_src)
                self.__streams[i_src['id']] = src

    def get_frame(self, src):
        self.__streams[src].f_grab()
        return self.__streams[src].f_retrieve()

    def get(self, src):
        return self.__streams[src]
