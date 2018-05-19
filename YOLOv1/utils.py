import cv2
import numpy as np


# bboxes shape = (num_objs, 4)
# bboxes[0] = (xmin, xmax, ymin, ymax)
def resize_to_448x448(img, bboxes):
    w = img.shape[1]
    h = img.shape[0]

    w_ratio = 448 / w
    h_ratio = 448 / h

    img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_CUBIC)
    bboxes[:, :2] *= w_ratio
    bboxes[:, 2:] *= h_ratio

    return img, bboxes


# bboxes shape = (num_objs, 4)
# bboxes[0] = (xmin, xmax, ymin, ymax)
def bbox_to_xywh(bboxes):
    xywh = np.zeros_like(bboxes)
    xywh[:, 0] = (bboxes[:, 0] + bboxes[:, 1]) / 2
    xywh[:, 1] = (bboxes[:, 2] + bboxes[:, 3]) / 2
    xywh[:, 2] = (bboxes[:, 1] - bboxes[:, 0]) / 448
    xywh[:, 3] = (bboxes[:, 3] - bboxes[:, 2]) / 448

    qx = xywh[:, 0] // 64
    rx = xywh[:, 0] % 64
    qy = xywh[:, 1] // 64
    ry = xywh[:, 1] % 64

    xywh[:, 0] = rx / 64
    xywh[:, 1] = ry / 64

    return qx, qy, xywh

