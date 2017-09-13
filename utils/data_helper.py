import os
import sys


def get_ucf101_split(split_dir, split_id):
    """
    This func helps to obtain the training data set and testing data set

    :returns
        - classes_labels: dict of {class_name: class_label}
        - train_videos_classes: dict of {train_video_name: class_name}
        - test_videos_classes: dict of {test_video_name: class_name}
    """
    classes_labels = read_label_file(split_dir + 'classInd.txt')
    train_file_path = split_dir + 'trainlist0%d'%(split_id) + '.txt'
    test_file_path = split_dir + 'testlist0%d'%(split_id) + '.txt'

    train_videos_classes = read_file(train_file_path)
    test_videos_classes = read_file(test_file_path)

    return classes_labels, train_videos_classes, test_videos_classes


def read_file(file_path):
    """
    The data format in the input_file should be like: 'ApplyEyeMakeup/v_ApplyEyeMakeup_g10_c02.avi 1'

    :return
        - videos_classes: dict of {video_name: class_name}
    """
    f = open(file_path, 'rU')
    f = filter(lambda x: x.find('.avi') != -1, f)
    #process conflict in this file with label file
    f = map(lambda x: x.replace('HandStandPushups', 'HandstandPushups'), f)

    videos_classes = {}
    for ef in f:
        class_name, video_name = (ef.split('.')[0]).split('/')
        videos_classes[video_name] = class_name
    return videos_classes


def read_label_file(label_file):
    """
    read the label document . The data format in the label_file is like: '2 ApplyLipstick'

    :returns
        - classes_labels: dict of {class_name: class_label}
    """
    label_file = open(label_file, 'rU')
    classes_labels = {}
    for line in label_file:
        line_list = line.replace('\n', '').split(' ')
        classes_labels[line_list[1]] = int(line_list[0])
    return classes_labels

