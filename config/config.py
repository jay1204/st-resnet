import numpy as np
from easydict import EasyDict as ed

ucf = ed()
ucf.split_dir = 'data/ucfTrainTestlist/'
ucf.split_id = 1
ucf.num_classes = 101

ucf.image = ed()
ucf.image.dir = 'data/ucf101_jpegs/'
ucf.image.data_shape = (224, 224, 3)

train_image = ed()
train_image.batch_size = 256
train_image.epoch = 10
train_image.drop_out = 0.0
# augmentation option: 'borders25', 'rand_crop'
train_image.augmentation = ('rand_crop')
train_image.n_frames_per_video = 1
train_image.learning_rate = 0.01


train_flow = ed()
train_flow.batch_size = 256
train_flow.epoch = 10
train_flow.drop_out = 1.0

resnet_50 = ed()
resnet_50.dir = 'data/pretrained_model/'
resnet_50.name = 'resnet-50'
resnet_50.model_epoch = 0
resnet_50.url_prefix = 'http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50'
resnet_50.data_shape = (3, 224, 224)














