import mxnet as mx
from config import ucf, train_image, resnet_50, test_image, vgg_16
from model import ConvNet
from utils import get_ucf101_split
import random
import logging
import time


def main():
    logging.basicConfig(filename='log/experiment_spatial.log', level=logging.INFO)
    logging.info("Start training image network: {}".format(time.asctime(time.localtime(time.time()))))
    ctx = mx.gpu(0)
    classes_labels, train_videos_classes, test_videos_classes = get_ucf101_split(ucf.split_dir, ucf.split_id)

    #videos = list(test_videos_classes.keys())
    #sample_videos= random.sample(videos, 500)
    #test_videos_classes_samples = {}
    #for video in sample_videos:
    #    test_videos_classes_samples[video] = test_videos_classes[video]

    cm = ConvNet(model_params=resnet_50, data_params=ucf.image, train_params=train_image, test_params=test_image,
                 train_videos_classes=train_videos_classes, test_videos_classes=test_videos_classes,
                 classes_labels=classes_labels, num_classes=ucf.num_classes, ctx=ctx, mode='spatial')
    cm.train()

    return

if __name__ == '__main__':
    main()
