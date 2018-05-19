import cv2
from YOLOv1.utils import resize_to_448x448, bbox_to_xywh
from utils.voc_parser import get_test, get_train, get_val
from utils.augmentation import augmentation
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
            bboxes = bboxes_list[i]
            classes = classes_list[i]
            img, bboxes = resize_to_448x448(img, bboxes)

            # augmentation
            if aug:
                img, bboxes = augmentation(img, bboxes)

            qx, qy, xywh = bbox_to_xywh(bboxes)




    y_imgs = []
    y_bboxes_list = []
    y_rpn_cls_list = []
    y_rpn_regr_list = []
    y_classes_list = []

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

            y_imgs.append(img)
            y_bboxes_list.append(bboxes)
            y_rpn_cls_list.append(y_rpn_cls)
            y_rpn_regr_list.append(y_rpn_regr)
            y_classes_list.append(classes)

            if len(y_classes_list) == batch_size:
                yield y_imgs, y_bboxes_list, y_rpn_cls_list, y_rpn_regr_list, y_classes_list
                y_imgs.clear()
                y_bboxes_list.clear()
                y_rpn_cls_list.clear()
                y_rpn_regr_list.clear()
                y_classes_list.clear()
