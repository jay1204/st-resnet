import os
import sys
import numpy as np
import csv
from config import ucf


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


def make_ucf_image_lst():
    """
    It creates the ucf_image_train.lst and ucf_image_test.lst
    :return: None
    """
    # create ucf_image_train.lst
    make_lst(instruction_file_path= ucf.split_dir + 'trainlist0%d'%ucf.split_id + '.txt', data_dir=ucf.image.dir,
             label_file=os.path.join(ucf.split_dir, 'classInd.txt'),
             output_file_path=ucf.split_dir + 'train0%d'%ucf.split_id + '_image.lst')
    make_lst(instruction_file_path= ucf.split_dir + 'testlist0%d'%ucf.split_id + '.txt', data_dir=ucf.image.dir,
             label_file=os.path.join(ucf.split_dir, 'classInd.txt'),
             output_file_path=ucf.split_dir + 'test0%d'%ucf.split_id + '_image.lst')
    return


def make_ucf_flow_lst():
    pass


def make_lst(instruction_file_path, data_dir, label_file, output_file_path):
    video_path_list, video_group_name = read_instruction_file(instruction_file_path, data_dir)
    label_dict = read_label_file(label_file)

    # given video group name and label_dict, retrieve labels for each item in video_path_list
    labels = map(lambda x: label_dict[x], video_group_name)
    write_success = write_to_file(output_file_path, video_path_list, labels)
    if not write_success:
        raise SyntaxError('Could not create {} file.'.format(output_file_path))
    return


def write_to_file(file_name, video_path_list, labels):
    file_handler = csv.writer(open(file_name, "w"), delimiter='\t', lineterminator='\n')
    image_list = []
    counter = 0
    for i, video_path in enumerate(video_path_list):
        for img in os.listdir(video_path):
            if img.endswith('jpg'):
                image_list.append((counter, labels[i], os.path.join(video_path, img)))
                counter += 1

    for il in image_list:
        file_handler.writerow(il)
    return True


def read_instruction_file(instruction_file_path, data_dir):
    """
    The data format in the instruction_file should be like:
    'ApplyEyeMakeup/v_ApplyEyeMakeup_g10_c02.avi 1'

    :returns:
        - video_path_list: list of video paths, like jpegs_256/v_ApplyEyeMakeup_g10_c02
        - video_group_name: list of video group name, like ApplyEyeMakeup
    """
    instruction_file = open(instruction_file_path, 'rU')
    input_file_info = filter(lambda x: x.find('.avi') != -1, instruction_file)
    # process conflict in InputFileInfo
    input_file_info = map(lambda x: x.replace('HandStandPushups', 'HandstandPushups'), input_file_info)
    video_path_list = map(lambda x: data_dir + (x.split('.')[0]).split('/')[-1], input_file_info)
    video_group_name = map(lambda x: (x.split('.')[0]).split('/')[-2], input_file_info)

    return video_path_list, video_group_name


def process_lst_file(lst_file_path):
    with open(lst_file_path) as fin:
        lst_dict = {}
        for line in iter(fin.readline, ''):
            line = line.strip().split('\t')
            key = int(line[0])
            lst_dict[line[-1]] = key

    return lst_dict
