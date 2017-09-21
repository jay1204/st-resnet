import mxnet as mx
from config import ucf, train_image, resnet_50, test_image
from model import ConvNet
from utils import get_ucf101_split
from logger import logger
import time


def main():
    logger.info("Start testing spatial network: ", time.asctime(time.localtime(time.time())))
    ctx = mx.gpu(0)
    classes_labels, train_videos_classes, test_videos_classes = get_ucf101_split(ucf.split_dir, ucf.split_id)

    cm = ConvNet(model_params=resnet_50, data_params=ucf.image, train_params=train_image, test_params=test_image,
                 train_videos_classes=train_videos_classes, test_videos_classes=test_videos_classes,
                 classes_labels=classes_labels, num_classes=ucf.num_classes, ctx=ctx)
    cm.test_dataset_evaluation()
    return

if __name__ == '__main__':
    main()
