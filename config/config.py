import numpy as np
from easydict import EasyDict as ed

ucf = ed()
ucf.split_dir = 'data/ucfTrainTestlist/'
ucf.split_id = 1
ucf.num_classes = 101

ucf.image = ed()
ucf.image.dir = 'data/jpegs_256/'
ucf.image.data_shape = (224, 224, 3)

train_image = ed()
train_image.batch_size = 90
train_image.epoch = 10
train_image.drop_out = 0.1
train_image.augmentation = ['random_horizon_flip', 'random_border25_crop']
train_image.n_frames_per_video = 1
train_image.learning_rate = 0.0005

test_image = ed()
# batch_size should be identical to frame_per_video for testing
test_image.batch_size = 25
test_image.frame_per_video = 25
test_image.augmentation = [[], ['horizon_flip']]

#test_image.augmentation = [['left_top_corner_crop'], ['left_bottom_corner_crop'], ['right_top_corner_crop'],
#                           ['right_bottom_corner_crop'], ['centre_crop'], ['left_top_corner_crop', 'horizon_flip'],
#                           ['left_bottom_corner_crop', 'horizon_flip'], ['right_top_corner_crop', 'horizon_flip'],
#                           ['right_bottom_corner_crop', 'horizon_flip'], ['centre_crop', 'horizon_flip']]

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














