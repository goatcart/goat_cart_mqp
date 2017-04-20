import cv2
import numpy as np

title = 'Goat Cart'
color = np.asarray([63, 63, 63], 'uint8')
textcolor = (255, 255, 255)
padding = 150
font = cv2.FONT_HERSHEY_SIMPLEX
text_scale = 1
text_thick = 2


def get_title_size(title):
    return cv2.getTextSize(title, font, text_scale, text_thick)[0]

def combine_frames(max_w, *frames):
    # Get a copy of each frame so that original is not disturbed
    f_img = list(map(lambda x: x[0].copy(), frames))
    max_w = int(max_w / len(f_img))
    max_w_f = max_w - padding
    f_sizes = list(map(lambda f: f.shape[:2], f_img))
    scales = list(map(lambda sz: min(float(max_w_f) / sz[1], 1), f_sizes))
    for i in range(len(f_img)):
        if scales[i] < 1:
            f_sizes[i] = [int(f_sizes[i][0] * scales[i]), int(f_sizes[i][1] * scales[i])]
            f_img[i] = cv2.resize(f_img[i], (f_sizes[i][1], f_sizes[i][0]))
        if len(f_img[i].shape) == 2:
            f_img[i] = cv2.cvtColor(f_img[i], cv2.COLOR_GRAY2RGB)
    size = [max(sizes) for sizes in zip(*f_sizes)]
    max_w = min(max_w, size[1] + padding)
    max_h = size[0] + padding
    frame = np.ones((max_h, max_w * len(f_img), 3), np.uint8) * color
    for i in range(len(f_img)):
        x = int((max_w - f_sizes[i][1]) / 2) + max_w * i
        y = int((max_h - f_sizes[i][0]) / 2)
        h, w = f_sizes[i]
        frame[ y : y + h, x : x + w ] = f_img[i]
        title_w, title_h = get_title_size(frames[i][1])
        title_loc = (x + int((w - title_w) / 2), max_h - title_h - 5)
        textcolor_ = textcolor
        if len(frames[i]) > 2:
            textcolor_ = tuple(frames[i][2])
        cv2.putText(frame, frames[i][1],
            title_loc,
            font, text_scale, textcolor_, text_thick,
            cv2.LINE_AA)
    title_w, title_h = get_title_size(title)
    cv2.putText(frame, title,
        (int((max_w * len(f_img) - title_w) / 2), 20 + title_h), font,
        text_scale, textcolor, text_thick,
        cv2.LINE_AA)
    return frame