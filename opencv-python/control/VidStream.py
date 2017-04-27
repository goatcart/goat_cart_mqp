import cv2
import numpy as np

__sides = ['left', 'right']

class SourceBase:
    """Base image source"""
    _img = None   # Loaded/captured image
    _shape = None # Image dimensions

    def __init__(self, i_src):
        self.__info = i_src

    def name(self):
        """Get source name"""
        if self.__info and 'name' in self.__info:
            return self.__info['name']
        return 'Unkown'

    def scale(self):
        """Get scaling applied to source"""
        if self.__info and 'scale' in self.__info:
            return self.__info['scale']
        return 1.0

    def size(self):
        """Get size"""
        if self._shape:
            return self._shape
        if self.__info and 'rez' in self.__info:
            return tuple(self.__info['rez'])
        return (480, 360)

    def update(self):
        """Update the image, for this, do nothing"""
        return

    def frames(self):
        return self._img

class ImageSource(SourceBase):
    """Image source is a static image file"""
    def __init__(self, i_src):
        super().__init__(i_src)
        path = i_src['path']
        # Load image(s)
        self._img = [cv2.imread(path + src, cv2.IMREAD_COLOR) for src in i_src['src']]
        # Get image shape, assume all images have the same dimensions
        self._shape = (self._img[0].shape[1], self._img[0].shape[0])

class VidSource(SourceBase):
    """Image source is a camera"""
    __src = [] # Array of input sources
    def __init__(self, i_src):
        super().__init__(i_src)
        for i in i_src['src']: # Connect to cameras
            cam = cv2.VideoCapture(i)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH,  i_src['rez'][0])
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, i_src['rez'][1])
            self.__src.append(cam) # Add to list of sources
        # Determine image size
        rez = self._SourceBase__info['rez']
        scale = self._SourceBase__info['scale']
        self._shape = (int(rez[1] * scale), int(rez[0] * scale))

    def update(self):
        """Grab new frames from camera"""
        # Tell camera to capture frames
        # This is done separately from actually retrieving the frames
        # For Synchronization purposes as the time taken to 'snap' a
        # Frame is short, unlike retrieving the captured frame
        for src in self.__src:
            src.grab()
        # Reset image list
        self._img = []
        scale = self.scale()
        # Retrieve + process frames
        for src in self.__src:
            # Retrieve frame from source
            ret, frame = src.retrieve()
            # Apply blur
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            # Scale if necessary
            if scale < 1.0:
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            # Append to list of frames
            self._img.append(frame)


class SourceManager:
    """Simple class to manage all input sources"""

    # Source dictionary: name -> Source
    __streams = {}
    def __init__(self, sources):
        # Save config data locally
        self.__sources = sources
        for source in sources: # Add all sources
            if source['enabled']:
                # By default, a dummy source
                src = SourceBase(None)
                if source['type'] == 'image': # Source is static image(s)
                    src = ImageSource(source)
                elif source['type'] == 'vid': # Source is camera
                    src = VidSource(source)
                # Add source
                self.__streams[source['id']] = src

    # Get source
    def get(self, src):
        return self.__streams[src]

    # Update sources
    def update(self):
        for src in self.__streams:
            self.__streams[src].update()
