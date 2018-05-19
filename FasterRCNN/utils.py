import numpy as np
import math
import random
import cv2


# define 9 anchor boxes
def _get_anchor_box():
    anchor_box = []
    anchor_box_scales = [128, 256, 512]
    anchor_box_ratios = [[1, 1], [1 / math.sqrt(2), math.sqrt(2)], [math.sqrt(2), 1 / math.sqrt(2)]]
    for i in range(3):
        for j in range(3):
            anchor_box.append([anchor_box_scales[i] * anchor_box_ratios[j][0],
                               anchor_box_scales[i] * anchor_box_ratios[j][1]])

    return np.array(anchor_box)


# resize img to min_pixel 600
def resize_img_600(img, bboxes):
    w = img.shape[1]
    h = img.shape[0]
    if w < h:
        rr = 600 / w
    else:
        rr = 600 / h

    img = cv2.resize(img, (math.ceil(w * rr), math.ceil(h * rr)), interpolation=cv2.INTER_CUBIC)
    bboxes = bboxes * rr

    return img, bboxes


# get iou. a, b should be (xmin, xmax, ymin, ymax)
def _get_iou(a, b):
    iw = min(a[1], b[1]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[2], b[2])

    if iw < 0 or ih < 0:
        intersection = 0
    else:
        intersection = iw * ih

    area_a = (a[1] - a[0]) * (a[3] - a[2])
    area_b = (b[1] - b[0]) * (b[3] - b[2])
    union = area_a + area_b - intersection

    return intersection / (union + 1e-6)


# get tx, ty, tw, th.
# a, b should be (xmin, xmax, ymin, ymax)
# a: ground_truth, b: default_box
def _calc_regr(a, b):
    gx = (a[0] + a[1]) / 2
    gy = (a[2] + a[3]) / 2
    gw = a[1] - a[0]
    gh = a[3] - a[2]

    px = (b[0] + b[1]) / 2
    py = (b[2] + b[3]) / 2
    pw = b[1] - b[0]
    ph = b[3] - b[2]

    tx = (gx - px) / (pw + 1e-6)
    ty = (gy - py) / (ph + 1e-6)
    tw = np.log(gw / (pw + 1e-6))
    th = np.log(gh / (ph + 1e-6))

    return [tx, ty, tw, th]


def get_rpn_ground_truth(img, bboxes):
    rpn_scale = 4

    anchor_boxes = _get_anchor_box()
    num_anchors = anchor_boxes.shape[0]

    # width, height of output feature map
    w, h = img.shape[1], img.shape[0]
    out_w = w // rpn_scale
    out_h = h // rpn_scale

    num_bbox = bboxes.shape[0]

    # valid(non neutral) anchors list
    # plays a role as a filter. It helps only positive or negative samples can affect loss.
    y_is_box_valid = np.zeros((out_h, out_w, num_anchors))

    # positive anchors list
    # check positive anchors as 1 and use binary cross entropy.
    y_rpn_overlap = np.zeros((out_h, out_w, num_anchors))

    # (tx, tw, tw, th) list.
    # for rpn regression loss
    y_rpn_regr = np.zeros((out_h, out_w, num_anchors, 4))

    # count positive for each object.
    # Use the highest iou anchor_box which object has no positive anchor.
    pos_count = np.zeros(num_bbox)

    # highest iou anchor_box per object lists.
    best_iou_per_obj = np.zeros(num_bbox)
    best_iou_index_per_obj = np.zeros((num_bbox, 3), dtype=np.int16)
    best_iou_regr_per_obj = np.zeros((num_bbox, 4))

    # check all anchor boxes for all output_feature_map pixels
    for ih in range(out_h):
        for iw in range(out_w):
            for ia, anc in enumerate(anchor_boxes):
                anchor_type = 'neutral'

                # projection to original image with out_feature_map.
                anc_xmin = (iw + 0.5) * rpn_scale - anc[0] / 2
                anc_xmax = (iw + 0.5) * rpn_scale + anc[0] / 2
                anc_ymin = (ih + 0.5) * rpn_scale - anc[1] / 2
                anc_ymax = (ih + 0.5) * rpn_scale + anc[1] / 2

                # Discard anchor_box that is not within the range of original image.
                if anc_xmin < 0 or anc_xmax > w or anc_ymin < 0 or anc_ymax > h:
                    continue

                # To find regr(tx,ty,tw,th) for the most appropriate object.
                max_iou_in_anchors = -1
                max_regr = [0, 0, 0, 0]

                for ib in range(num_bbox):
                    # Get iou with all bboxes
                    cur_iou = _get_iou(bboxes[ib], [anc_xmin, anc_xmax, anc_ymin, anc_ymax])

                    if cur_iou > 0.7:
                        anchor_type = 'pos'
                        pos_count[ib] += 1
                    elif cur_iou < 0.3:
                        if anchor_type != 'pos':
                            anchor_type = 'neg'

                    # if this bbox is the most appropriate object for anchor (max_iou)
                    if cur_iou > max_iou_in_anchors:
                        max_regr = _calc_regr(bboxes[ib], [anc_xmin, anc_xmax, anc_ymin, anc_ymax])

                    # check highest iou per object
                    if cur_iou > best_iou_per_obj[ib]:
                        best_iou_per_obj[ib] = cur_iou
                        best_iou_index_per_obj[ib] = [ih, iw, ia]
                        best_iou_regr_per_obj[ib] = _calc_regr(bboxes[ib], [anc_xmin, anc_xmax, anc_ymin, anc_ymax])

                if anchor_type == 'neutral':
                    y_is_box_valid[ih, iw, ia] = 0
                    y_rpn_overlap[ih, iw, ia] = 0
                elif anchor_type == 'neg':
                    y_is_box_valid[ih, iw, ia] = 1
                    y_rpn_overlap[ih, iw, ia] = 0
                elif anchor_type == 'pos':
                    y_is_box_valid[ih, iw, ia] = 1
                    y_rpn_overlap[ih, iw, ia] = 1
                    y_rpn_regr[ih, iw, ia] = max_regr

    # watch pos_count and use highest iou anchor_box for 0_counted object.
    for ib in range(num_bbox):
        if pos_count[ib] == 0:
            [ih, iw, ia] = best_iou_index_per_obj[ib]
            y_is_box_valid[ih, iw, ia] = 1
            y_rpn_overlap[ih, iw, ia] = 1
            y_rpn_regr[ih, iw, ia] = best_iou_regr_per_obj

    # get all positive anchors' indexes
    pos_locs = np.where(np.logical_and(y_is_box_valid == 1, y_rpn_overlap == 1))
    neg_locs = np.where(np.logical_and(y_is_box_valid == 1, y_rpn_overlap == 0))

    data_limit = 256
    num_pos = len(pos_locs[0])

    # case 1: num_pos > 128
    if len(pos_locs[0]) > data_limit // 2:
        # random sample (num_pos - 128) indexes.
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - data_limit // 2)

        # sampled indexes --> not valid
        y_is_box_valid[pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0

        num_pos = data_limit // 2

    # pos == neg
    if len(neg_locs[0]) + num_pos > data_limit:
        # random sample indexes.
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)

        # sampled indexes --> not valid
        y_is_box_valid[neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=2)
    y_rpn_regr = y_rpn_regr.reshape((out_h, out_w, num_anchors * 4))
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=2), y_rpn_regr], axis=2)

    return y_rpn_cls, y_rpn_regr
