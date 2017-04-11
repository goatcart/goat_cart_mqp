import cv2
import numpy as np

__sides = ['left', 'right']

class SourceBase:
    _img = None
    _shape = None

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
        if self._shape:
            return self._shape
        if self.__info and 'rez' in self.__info:
            return tuple(self.__info['rez'])
        return (480, 360)

    def update(self):
        return

    def frames(self):
        return self._img

class ImageSource(SourceBase):
    def __init__(self, i_src):
        super().__init__(i_src)
        path = i_src['path']
        self._img = [cv2.imread(path + src, cv2.IMREAD_COLOR) for src in i_src['src']]
        self._shape = (self._img[0].shape[1], self._img[0].shape[0])

### Yeah, needs to be improved
class VidSource(SourceBase):
    __src = []
    def __init__(self, i_src):
        super().__init__(i_src)
        for i in i_src['src']:
            cam = cv2.VideoCapture(i)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH,  i_src['rez'][0])
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, i_src['rez'][1])
            self.__src.append(cam)
            rez = self._SourceBase__info['rez']
            scale = self._SourceBase__info['scale']
            self._shape = (int(rez[1] * scale), int(rez[0] * scale))

    def update(self):
        for src in self.__src:
            src.grab()
        success = True
        self._img = []
        scale = self.scale()
        for src in self.__src:
            ret, frame = src.retrieve()
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            if scale < 1.0:
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            self._img.append(frame)


class SourceManager:
    __streams = {}
    def __init__(self, sources):
        self.__sources = sources
        for source in sources:
            if source['enabled']:
                src = SourceBase(None)
                if source['type'] == 'image':
                    src = ImageSource(source)
                elif source['type'] == 'vid':
                    src = VidSource(source)
                self.__streams[source['id']] = src

    def get(self, src):
        return self.__streams[src]

    def update(self):
        for src in self.__streams:
            self.__streams[src].update()
