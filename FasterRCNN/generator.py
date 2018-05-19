import numpy as np
import cv2

from FasterRCNN.utils import get_rpn_ground_truth, resize_img_600
from utils.augmentation import augmentation
from utils.voc_parser import get_train, get_val, get_test
from sklearn.utils import shuffle


def voc_generator(voc_path, mode, batch_size):
    if mode == 'train':
        img_paths, bboxes_list, classes_list = get_train(voc_path)
        aug = True
    elif mode == 'val':
        img_paths, bboxes_list, classes_list = get_val(voc_path)
        aug = True
    elif mode == 'test':
        img_paths, bboxes_list, classes_list = get_test(voc_path)
        aug = False
    else:
        return

    num_data = img_paths.shape[0]

    while True:
        img_paths, bboxes_list, classes_list = shuffle(img_paths, bboxes_list, classes_list)

        for i in range(num_data):
            img = cv2.imread(img_paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bboxes = np.array(bboxes_list[i])
            classes = np.array(classes_list[i])

            # augmentation
            if aug:
                img, bboxes = augmentation(img, bboxes)

            # resize img
            img, bboxes = resize_img_600(img, bboxes)

            # get rpn GT
            y_rpn_cls, y_rpn_regr = get_rpn_ground_truth(img, bboxes)

            yield img, bboxes, y_rpn_cls, y_rpn_regr, classes
