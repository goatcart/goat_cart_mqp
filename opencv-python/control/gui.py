import cv2
import numpy as np

title = 'Goat Cart'
frame_titles = ['1', '2', '3', '4', '5']
color = np.asarray([63, 63, 63], 'uint8')
textcolor = (255, 255, 255)
padding = 150
font = cv2.FONT_HERSHEY_SIMPLEX
text_scale = 1
text_thick = 2


def get_title_size(title):
    return cv2.getTextSize(title, font, text_scale, text_thick)[0]

def combine_frames(frames, max_w):
    # Get a copy of each frame so that original is not disturbed
    frames = list(map(lambda x: x.copy(), frames))
    max_w = int(max_w / len(frames))
    max_w_f = max_w - padding
    f_sizes = list(map(lambda f: f.shape[:2], frames))
    scales = list(map(lambda sz: min(max_w_f / sz[1], 1), f_sizes))
    for i in range(len(frames)):
        if scales[i] < 1:
            f_sizes[i] = [int(f_sizes[i][0] * scales[i]), int(f_sizes[i][1] * scales[i])]
            frames[i] = cv2.resize(frames[i], (f_sizes[i][1], f_sizes[i][0]))
        if len(frames[i].shape) == 2:
            frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2RGB)
    size = [max(sizes) for sizes in zip(*f_sizes)]
    max_w = min(max_w, size[1] + padding)
    max_h = size[0] + padding
    frame = np.ones((max_h, max_w * len(frames), 3), np.uint8) * color
    for i in range(len(frames)):
        x = int((max_w - f_sizes[i][1]) / 2) + max_w * i
        y = int((max_h - f_sizes[i][0]) / 2)
        h, w = f_sizes[i]
        frame[ y : y + h, x : x + w ] = frames[i]
        title_w, title_h = get_title_size(frame_titles[i])
        title_loc = (x + int((w - title_w) / 2), max_h - title_h - 5)
        cv2.putText(frame, frame_titles[i],
            title_loc,
            font, text_scale, textcolor, text_thick,
            cv2.LINE_AA)
    title_w, title_h = get_title_size(title)
    cv2.putText(frame, title,
        (int((max_w * len(frames) - title_w) / 2), 20 + title_h), font,
        text_scale, textcolor, text_thick,
        cv2.LINE_AA)
    return frame