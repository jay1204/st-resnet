import mxnet as mx
from config import ucf, train_image, resnet_50, test_image
from model import ConvImage
from utils import get_ucf101_split

def main():
    ctx = mx.gpu(0)
    classes_labels, train_videos_classes, test_videos_classes = get_ucf101_split(ucf.split_dir, ucf.split_id)

    videos = list(train_videos_classes.keys())[:100]
    train_videos_classes_sample = {}
    for video in videos:
        train_videos_classes_sample[video] = train_videos_classes[video]

    cm = ConvImage(model_params=resnet_50, data_params=ucf.image, train_params=train_image, test_params=test_image,
                   train_videos_classes=train_videos_classes_sample, test_videos_classes=test_videos_classes,
                   classes_labels=classes_labels, num_classes=ucf.num_classes, ctx=ctx)
    cm.train()
    return

if __name__ == '__main__':
    main()
