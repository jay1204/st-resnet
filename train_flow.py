import mxnet as mx
from config import ucf, train_image, resnet_50, test_image, train_flow, test_flow
from model import ConvNet
from utils import get_ucf101_split
import random
import logging
import time
import math


def main():
    logging.basicConfig(filename='log/experiment_temporal.log', level=logging.INFO)
    logging.info("Start training flow network: {}".format(time.asctime(time.localtime(time.time()))))
    ctx = [mx.gpu(0), mx.gpu(1)]
    #ctx = [mx.gpu(0)]
    classes_labels, train_videos_classes, test_videos_classes = get_ucf101_split(ucf.split_dir, ucf.split_id)

    #videos = list(test_videos_classes.keys())
    #sample_videos= random.sample(videos, 500)
    #test_videos_classes_samples = {}
    #for video in sample_videos:
    #    test_videos_classes_samples[video] = test_videos_classes[video]

    cm = ConvNet(model_params=resnet_50, data_params=ucf.flow, train_params=train_flow, test_params=test_flow,
                     train_videos_classes=train_videos_classes, test_videos_classes=test_videos_classes,
                     classes_labels=classes_labels, num_classes=ucf.num_classes, ctx=ctx, mode='temporal')
    cm.train()
    return

if __name__ == '__main__':
    main()
