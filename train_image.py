import mxnet as mx
from config import ucf, train_image, resnet_50
from model import ConvImage
from utils import get_ucf101_split

def main():
    ctx = mx.gpu(0)
    classes_labels, train_videos_classes, test_videos_classes = get_ucf101_split(ucf.split_dir, ucf.split_id)

    cm = ConvImage(model_params=resnet_50, data_params=ucf.image, train_params=train_image,
                   train_videos_classes=train_videos_classes, test_videos_classes=test_videos_classes,
                   classes_labels=classes_labels, ctx=ctx)
    return

if __name__ == '__main__':
    main()