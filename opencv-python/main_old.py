import numpy as np
import cv2

'''
normalize(data->cv_image_disparity, data->cv_image_disparity_normalized, 0,
    255, CV_MINMAX, CV_8UC1);
cvtColor(data->cv_image_disparity_normalized, data->cv_color_image,
    CV_GRAY2RGB);
GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(
    (guchar*) data->cv_color_image.data, GDK_COLORSPACE_RGB, false,
    8, data->cv_color_image.cols,
    data->cv_color_image.rows, data->cv_color_image.step,
    NULL, NULL);
'''

print cv2.__version__
sz = (960,720)
scale = 1.0
FPS  = 5
cap_l = cv2.VideoCapture(1)
cap_r = cv2.VideoCapture(2)
cap_l.set(cv2.CAP_PROP_FRAME_WIDTH, int(sz[0] * scale))
cap_l.set(cv2.CAP_PROP_FRAME_HEIGHT, int(sz[1] * scale))
cap_l.set(cv2.CAP_PROP_FPS, FPS)
cap_r.set(cv2.CAP_PROP_FRAME_WIDTH, int(sz[0] * scale))
cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT, int(sz[1] * scale))
cap_r.set(cv2.CAP_PROP_FPS, FPS)

bs = 5
numD = ((sz[0]/8) + 15) & -16
P_LCM = 3 * bs * bs

stereo = cv2.StereoSGBM_create(
    blockSize         = bs,
    minDisparity      = 0,
    numDisparities    = numD,
    disp12MaxDiff     = 10,
    speckleRange      = 7,
    speckleWindowSize = 60,
    P1                = 8 * P_LCM,
    P2                = 32 * P_LCM,
    preFilterCap      = 1,
    uniquenessRatio   = 0,
    mode              = 1
)

first=True

while(True):
    # Capture frame-by-frame
    ret_l, frame_l = cap_l.read()
    assert ret_l
    ret_r, frame_r = cap_r.read()
    assert ret_r

    if first:
        cv2.imwrite('../stereo-tuner/lab/frame_l.png', frame_l)
        cv2.imwrite('../stereo-tuner/lab/frame_r.png', frame_r)
        first=False

    frame_l_g = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
    frame_r_g = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(frame_l_g, frame_r_g)
    disparity_norm = np.ndarray(disparity.shape, dtype=np.uint8)
    cv2.normalize(disparity, disparity_norm,
        alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U)
    disparity_color = cv2.cvtColor(disparity_norm, cv2.COLOR_GRAY2RGB)

    # Display the resulting frame
    cv2.imshow('left', frame_l)
    cv2.imshow('right', frame_r)
    cv2.imshow('disparity', disparity_color)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap_l.release()
cap_r.release()
cv2.destroyAllWindows()
