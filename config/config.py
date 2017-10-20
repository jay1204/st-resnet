import numpy as np
from easydict import EasyDict as ed
import os

ucf = ed()
ucf.split_dir = 'data/ucfTrainTestlist/'
ucf.split_id = 1
ucf.num_classes = 101

ucf.image = ed()
ucf.image.dir = ['data/jpegs_256/']
ucf.image.data_shape = (224, 224, 3)
ucf.image.lst_file = ucf.split_dir + 'train0%d'%ucf.split_id + '_image.lst'
#ucf.image.rec_file = ucf.split_dir + 'train0%d'%ucf.split_id + '_image.rec'
#ucf.image.idx_file = None
#ucf.image.idx_file = ucf.split_dir + 'train0%d'%ucf.split_id + '_image.idx'

ucf.flow = ed()
ucf.flow.data_shape = (224, 224, 3)
ucf.flow.dir = ['data/tvl1_flow/u', 'data/tvl1_flow/v']

train_image = ed()
train_image.batch_size = 90
#train_image.epoch = 10
train_image.drop_out = 0.9
train_image.augmentation = ['random_horizon_flip', 'random_border25_crop']
train_image.clip_per_video = 1
train_image.learning_rate = 1e-4#1e-5
train_image.resume = False
train_image.load_epoch = 8
train_image.iteration = 30000
train_image.schedule_steps = [10000, 20000, 30000]
train_image.use_global_stats = True
train_image.frame_per_clip = 1

test_image = ed()
# batch_size should be identical to frame_per_video for testing
test_image.batch_size = 25
test_image.clip_per_video = 25
test_image.augmentation = [[], ['horizon_flip']]
test_image.load_epoch = 6
test_image.remove_softmax_layer = True


#test_image.augmentation = [['left_top_corner_crop'], ['left_bottom_corner_crop'], ['right_top_corner_crop'],
#                           ['right_bottom_corner_crop'], ['centre_crop'], ['left_top_corner_crop', 'horizon_flip'],
#                           ['left_bottom_corner_crop', 'horizon_flip'], ['right_top_corner_crop', 'horizon_flip'],
#                           ['right_bottom_corner_crop', 'horizon_flip'], ['centre_crop', 'horizon_flip']]

train_flow = ed()
train_flow.batch_size = 70
#train_flow.epoch = 10
train_flow.drop_out = 0.8
train_flow.augmentation = ['random_horizon_flip', 'random_corner_crop']
train_flow.clip_per_video = 1
train_flow.learning_rate = 0.01
train_flow.resume = False
train_flow.load_epoch = 1
train_flow.iteration = 20000
train_flow.frame_per_clip = 10
train_flow.schedule_steps = [10000, 15000, 20000]
train_flow.use_global_stats = True

test_flow = ed()
test_flow.batch_size = 25
test_flow.clip_per_video = 25
test_flow.augmentation = [[], ['horizon_flip']]
test_flow.load_epoch = 1
test_flow.frame_per_clip = 10

resnet_50 = ed()
resnet_50.dir = 'data/pretrained_model/'
resnet_50.name = 'resnet-50'
resnet_50.model_epoch = 0
resnet_50.url_prefix = 'http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50'
resnet_50.data_shape = (3, 224, 224)

vgg_16=ed()
vgg_16.dir = 'data/pretrained_model/'
vgg_16.name = 'vgg16'
vgg_16.model_epoch = 0
vgg_16.url_prefix = 'http://data.dmlc.ml/models/imagenet/vgg/vgg16'
vgg_16.data_shape = (3, 224, 224)














