import os
import numpy as np
import xml.etree.ElementTree as ET


def _get_labels():
    return ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'background']


# get data from 'year' information
# mode can be train, val, trainval etc.. (only the name of voc category)
def _get_data(voc_path, year, mode):
    labels = _get_labels()

    data_path = os.path.join(voc_path, 'VOC' + year)
    annot_folder = os.path.join(data_path, 'Annotations')
    img_folder = os.path.join(data_path, 'JPEGImages')

    img_paths = []
    bboxes_list = []   # information of bounding boxes.
    classes_list = []  # class data

    # read .txt file and get image file names
    with open(os.path.join(data_path, 'ImageSets', 'Main', mode + '.txt')) as f:
        file_names = f.read().splitlines()

    # parse xml files
    for file_name in file_names:
        annot = os.path.join(annot_folder, file_name + '.xml')
        et = ET.parse(annot)
        element = et.getroot()

        # find all bojects
        element_objs = element.findall('object')

        bboxes = []
        classes = []
        for element_obj in element_objs:
            class_name = element_obj.find('name').text
            c_num = labels.index(class_name)

            # if difficulty is 1(hard to find object), skip it
            difficulty = int(element_obj.find('difficult').text)
            if difficulty:
                continue

            obj_bbox = element_obj.find('bndbox')
            xmin = int(round(float(obj_bbox.find('xmin').text)))
            ymin = int(round(float(obj_bbox.find('ymin').text)))
            xmax = int(round(float(obj_bbox.find('xmax').text)))
            ymax = int(round(float(obj_bbox.find('ymax').text)))

            bboxes.append([xmin, xmax, ymin, ymax])
            classes.append(c_num)

        # If there is no bboxes in the image, skip it (because we skip all difficulty==1 bboxes)
        if len(bboxes) > 0:
            img_paths.append(os.path.join(img_folder, file_name + '.jpg'))
            bboxes_list.append(bboxes)
            classes_list.append(classes)

    return np.array(img_paths), np.array(bboxes_list), np.array(classes_list)


# get train data
def get_train(voc_path):
    train2007_img_paths, train2007_bboxes_list, train2007_classes_list = _get_data(voc_path, '2007', 'train')
    train2012_img_paths, train2012_bboxes_list, train2012_classes_list = _get_data(voc_path, '2012', 'trainval')

    # concatenate informs
    train_img_paths = np.concatenate([train2007_img_paths, train2012_img_paths])
    train_bboxes_list = np.concatenate([train2007_bboxes_list, train2012_bboxes_list])
    train_classes_list = np.concatenate([train2007_classes_list, train2012_classes_list])

    return train_img_paths, train_bboxes_list, train_classes_list


# get val data
def get_val(voc_path):
    val2007_img_paths, val2007_bboxes_list, val2007_classes_list = _get_data(voc_path, '2007', 'val')

    return val2007_img_paths, val2007_bboxes_list, val2007_classes_list


# get test data
def get_test(voc_path):
    test2007_img_paths, test2007_bboxes_list, test2007_classes_list = _get_data(voc_path, '2007', 'test')

    return test2007_img_paths, test2007_bboxes_list, test2007_classes_list
