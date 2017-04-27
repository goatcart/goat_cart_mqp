import cv2
import numpy as np

title = 'Goat Cart'                        # Title
color = np.asarray([63, 63, 63], 'uint8')  # Background color
textcolor = (255, 255, 255)                # Default color for all text
padding = 150                              # Padding around frame images
font = cv2.FONT_HERSHEY_SIMPLEX            # Text font
text_scale = 1                             # Text size
text_thick = 2                             # Thickness of text curves


def get_title_size(title):
    return cv2.getTextSize(title, font, text_scale, text_thick)[0]

def combine_frames(max_w, *frames):
    """Combine the passed frames into a single frame that is at most
    `max_w` pixels in width.

    Args:
        max_w   (int): Maximum overall width
        frames (list): List of frames
            element: (numpy.ndarray, str  [,  (int, int, int)])
                     (frame,         title[,   title color   ])
    """
    # Get a copy of each frame so that original is not disturbed
    f_img = list(map(lambda x: x[0].copy(), frames))
    # Convert to maximum width per frame (+ padding)
    max_w = int(max_w / len(f_img))
    # Maximum frame width (no padding)
    max_w_f = max_w - padding
    # Get sizes of frames
    f_sizes = list(map(lambda f: f.shape[:2], f_img))
    # Get needed scaling factor for each frame
    scales = list(map(lambda sz: min(float(max_w_f) / sz[1], 1), f_sizes))
    # Process the frames
    for i in range(len(f_img)):
        if scales[i] < 1: # Needs to be scaled
            # Update frame size
            f_sizes[i] = [int(f_sizes[i][0] * scales[i]), int(f_sizes[i][1] * scales[i])]
            # Scale
            f_img[i] = cv2.resize(f_img[i], (f_sizes[i][1], f_sizes[i][0]))
        if len(f_img[i].shape) == 2: # Frame is grayscale, needs to be converted to RGB
            f_img[i] = cv2.cvtColor(f_img[i], cv2.COLOR_GRAY2RGB)
    # Get maximum width and height among frames to combine
    size = [max(sizes) for sizes in zip(*f_sizes)]
    # Determine uniform size for frame cells
    max_w = min(max_w, size[1] + padding)
    max_h = size[0] + padding
    # Create overall frame
    frame = np.ones((max_h, max_w * len(f_img), 3), np.uint8) * color
    # Blit all frames + render captions
    for i in range(len(f_img)):
        # Frame position
        x = int((max_w - f_sizes[i][1]) / 2) + max_w * i
        y = int((max_h - f_sizes[i][0]) / 2)
        h, w = f_sizes[i] # Frame size
        # Blit frame
        frame[ y : y + h, x : x + w ] = f_img[i]
        # Get caption dimensions
        title_w, title_h = get_title_size(frames[i][1])
        # Get caption render location
        title_loc = (x + int((w - title_w) / 2), max_h - title_h - 5)
        # Default text color can optionally be overridden
        textcolor_ = textcolor
        if len(frames[i]) > 2:
            textcolor_ = tuple(frames[i][2])
        # Render text
        cv2.putText(frame, frames[i][1],
            title_loc,
            font, text_scale, textcolor_, text_thick,
            cv2.LINE_AA)
    # Render Frame title
    title_w, title_h = get_title_size(title)
    cv2.putText(frame, title,
        (int((max_w * len(f_img) - title_w) / 2), 20 + title_h), font,
        text_scale, textcolor, text_thick,
        cv2.LINE_AA)
    return frame