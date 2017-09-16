import numpy as np
import os
import random
import mxnet as mx
from mxnet.executor_manager import _split_input_slice
import mxnet.ndarray as nd
from mxnet.image import *
from utils import horizon_flip


class ImageIter(mx.io.DataIter):
    """
    This class is a wrapper of the basic mx.io.DataIter.
    """
    def __init__(self, batch_size, data_shape, data_dir, videos_classes, classes_labels, ctx=None, data_name='data',
                 label_name='label', mode='train', augmentation=None):
        """

        :param batch_size:
        :param data_shape: tuple of the input shape of pretrained model of format (channels, height, weight)
        :param data_dir:
        :param videos_classes:
        :param classes_labels:
        :param ctx:
        :param data_name:
        :param label_name:
        :param mode:
        :param augmentation: tuple of string, each string indicating one augmentation operation
        """
        super(ImageIter, self).__init__()

        self.batch_size = batch_size
        self.mode = mode
        self.data_dir = data_dir
        self.videos_classes = videos_classes
        self.classes_labels = classes_labels
        self.augmentation = augmentation

        self.videos = np.asarray(list(videos_classes.keys()))

        if not isinstance(data_shape, tuple):
            data_shape = tuple(data_shape)
        self.data_shape = data_shape
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]

        self.data_name = data_name
        self.label_name = label_name

        self.provide_data = [(data_name, (batch_size, ) + data_shape)]
        self.provide_label = [(label_name, (batch_size, ))]

        self.cur = 0
        self.reset()

    def reset(self):
        self.cur = 0

    def iter_next(self):
        #return self.cur + self.batch_size <= self.img_size
        return True

    def next(self):
        if self.iter_next():
            batch_data, batch_label = self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=[batch_data], label=[batch_label], pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def get_batch(self):
        if self.mode == 'train':
            video_indices = np.random.choice(self.videos.shape[0], size=self.batch_size, replace=False)
            sample_videos = self.videos[video_indices]
            batch_data, batch_label = self.read_frames(sample_videos)

        return batch_data, batch_label

    def read_frames(self, sample_videos):
        """
        Read a series of frames by sampling one frame from each video.

        :param sample_videos: numpy array of video name
        :return:
        """
        c, h, w = self.data_shape
        batch_data = nd.empty((self.batch_size, c, h, w))
        batch_label = nd.empty(self.batch_size)
        for i in xrange(sample_videos.shape[0]):
            video_path = os.path.join(self.data_dir, sample_videos[i], '')
            frames = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
            sample_frame_name = random.choice(frames)
            sample_frame = self.next_image(os.path.join(video_path, sample_frame_name))
            if sample_frame.shape != self.data_shape:
                raise AssertionError('The size of the image is not matched with the required data_shape!')
            batch_data[i][:] = sample_frame
            batch_label[i][:] = self.classes_labels[self.videos_classes[sample_videos[i]]]

        return batch_data, batch_label

    def getpad(self):
        #if self.cur + self.batch_size > self.img_size:
        #    return self.cur + self.batch_size - self.img_size
        #else:
        return 0

    def getindex(self):
        return self.cur/self.batch_size

    def pre_process_image(self, image):
        """Transforms input data with specified augmentation."""
        #print type(self.augmentation)
        c, h, w = self.data_shape
        image_h, image_w, _ = image.shape
        for process in self.augmentation:
            #print process
            if process == 'rand_crop':
                image, _ = mx.img.random_crop(image, (w, h))
            elif process == 'horizon_flip':
                image = horizon_flip(image)
            elif process == 'corner_crop':
                rw = np.random.randint(low=224, high=257)
                rh = np.random.randint(low=168, high=193)
                center_crop = np.random.randint(2)
                if center_crop:
                    image, _ = mx.img.center_crop(image, (rw, rh))
                else:
                    which_corner = np.random.randint(4)
                    # left upper corner
                    if which_corner == 0:
                        image = mx.img.fixed_crop(image, x0=0, y0=0, w=rw, h=rh)
                    # right upper corner
                    elif which_corner == 1:
                        image = mx.img.fixed_crop(image, x0=0, y0=image_h-rh, w=rw, h=rh)
                    elif which_corner == 2:
                        image = mx.img.fixed_crop(image, x0=image_w-rw, y0=0, w=rw, h=rh)
                    else:
                        image = mx.img.fixed_crop(image, x0=image_w-rw, y0=image_h-rh, w=rw, h=rh)
                image = mx.img.imresize(image, w, h)

        return image

    def next_image(self, img_path):
        image = self.load_one_image(img_path)
        image = self.pre_process_image(image)
        image = self.post_process_image(image)

        return image

    @staticmethod
    def load_one_image(img_path):
        with open(img_path, 'rb') as fp:
            image_info = fp.read()

        return mx.img.imdecode(image_info)

    @staticmethod
    def post_process_image(image):
        """
        Transform the image to make it shape as (channel, height, width)
        """
        return nd.transpose(image, axes=(2, 0, 1))



