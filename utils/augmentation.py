import random
import cv2
import numpy as np


# random horizontally flip (50% probability)
def _rand_horizontal_flip(img, bboxes):
    w = img.shape[1]
    rand_flip = random.random()
    if rand_flip > 0.5:
        img = cv2.flip(img, flipCode=1)
        tmp = np.copy([bboxes[:, 0]])
        bboxes[:, 0] = w - bboxes[:, 1]
        bboxes[:, 1] = w - tmp[0]

    return img, bboxes


# random exposure and saturation (50% +-)
def _rand_sv(img):
    rand_s = random.uniform(0.5, 1.5)
    rand_v = random.uniform(0.5, 1.5)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tmp = np.ones_like(img[:, :, 1]) * 255
    img[:, :, 1] = np.where(img[:, :, 1] * rand_s > 255, tmp, img[:, :, 1] * rand_s)
    img[:, :, 2] = np.where(img[:, :, 2] * rand_v > 255, tmp, img[:, :, 2] * rand_v)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    return img


# random scale (10% +-)
def _rand_scale(img, bboxes):
    orig_w = img.shape[1]
    orig_h = img.shape[0]

    # resize img
    scale = random.uniform(0.9, 1.1)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    w = img.shape[1]
    h = img.shape[0]

    # bbox post processing after scaling
    # case1: rand_scale > 1.0
    if scale >= 1.0:
        # resize bbox
        bboxes[:, :2] = bboxes[:, :2] * scale - (w - orig_w) / 2
        bboxes[:, 2:] = bboxes[:, 2:] * scale - (h - orig_h) / 2

        # crop img
        img = img[(h - orig_h) // 2:(h + orig_h) // 2, (w - orig_w) // 2:(w + orig_w) // 2, :]

        # crop bbox
        problem_idxs = []
        for i in range(bboxes.shape[0]):
            # get bbox area before scale.
            pre_bbox_area = (bboxes[i][1] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][2])

            # bbox processing (if bbox crosses over the image boundary.)
            for j in range(4):
                if bboxes[i][j] < 0:
                    bboxes[i][j] = 0
                elif bboxes[i][j] > orig_w and j < 2:
                    bboxes[i][j] = orig_w
                elif bboxes[i][j] > orig_h and j >= 2:
                    bboxes[i][j] = orig_h

            # get bbox area after bbox processing
            post_bbox_area = (bboxes[i][1] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][2])

            # if the bbox is cut offed and the information lost is more than 10%, there is problem.
            if post_bbox_area / pre_bbox_area < 0.9:
                problem_idxs.append(i)

        # delete problem_idxs
        bboxes = np.delete(bboxes, problem_idxs, axis=0)

    # case2: rand_scale <= 1.0
    else:
        # resize bbox
        bboxes[:, :2] = bboxes[:, :2] * scale + (orig_w - w) // 2
        bboxes[:, 2:] = bboxes[:, 2:] * scale + (orig_h - h) // 2

        # zero padding img
        img = cv2.copyMakeBorder(img, (orig_h - h) // 2, (orig_h - h) // 2, (orig_w - w) // 2, (orig_w - w) // 2,
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return img, bboxes


# random transform (20% +-)
def _rand_transform(img, bboxes):
    w = img.shape[1]
    h = img.shape[0]

    # transform img
    rand_trans_x = int(random.uniform(-0.2, 0.2) * w)
    rand_trans_y = int(random.uniform(-0.2, 0.2) * h)
    m = np.float32([[1, 0, rand_trans_x], [0, 1, rand_trans_y]])
    img = cv2.warpAffine(img, m, (w, h))

    # transform bbox
    bboxes[:, :2] += rand_trans_x
    bboxes[:, 2:] += rand_trans_y

    # crop bbox
    problem_idxs = []
    for i in range(bboxes.shape[0]):
        # get bbox area before scale.
        pre_bbox_area = (bboxes[i][1] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][2])

        # bbox processing (if bbox crosses over the image boundary.)
        for j in range(4):
            if bboxes[i][j] < 0:
                bboxes[i][j] = 0
            elif bboxes[i][j] > w and j < 2:
                bboxes[i][j] = w
            elif bboxes[i][j] > h and j >= 2:
                bboxes[i][j] = h

        # get bbox area after bbox processing
        post_bbox_area = (bboxes[i][1] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][2])

        # if the bbox is cut offed and the information lost is more than 10%, there is problem.
        if post_bbox_area / pre_bbox_area < 0.9:
            problem_idxs.append(i)

    # delete problem_idxs
    bboxes = np.delete(bboxes, problem_idxs, axis=0)

    return img, bboxes


# random vertical flip + random exposure and saturation
def augmentation(img, bboxes):
    img, bboxes = _rand_horizontal_flip(img, bboxes)
    img = _rand_sv(img)
    img, bboxes = _rand_scale(img, bboxes)
    img, bboxes = _rand_transform(img, bboxes)

    return img, bboxes
