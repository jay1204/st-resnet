import mxnet as mx
import numpy as np
import multiprocessing, threading
import logging
from utils.image_process import load_one_image, post_process_image, pre_process_image
import mxnet.ndarray as nd
import os


class VideoIter(mx.io.DataIter):
    """
    This class is a wrapper of the basic mx.io.DataIter.
    """
    def __init__(self, batch_size, data_shape, data_dir, videos_classes, classes_labels, ctx=None, data_name='data',
                 label_name='label', augmentation=None, frame_per_clip=1, shuffle=False):
        """
        :param batch_size:
        :param data_shape: tuple of the input shape of pretrained model of format (channels, height, weight)
        :param data_dir: list of strings
        :param videos_classes:
        :param classes_labels:
        :param ctx:
        :param data_name:
        :param label_name:
        :param mode: string, indicating whehter it is in the training phrase or test phrase
        :param augmentation: list of list, each list has strings indicating augmentation operations
        :param record: rec file type
        :param lst_dict: dict of key is string of the frame directory, value is the index in .lst file
        """
        super(VideoIter, self).__init__()

        self.batch_size = batch_size

        if not isinstance(data_dir, list) or len(data_dir) <= 0:
            raise SyntaxError('The data_dir should be a list and contains at least some data directory')
        self.data_dir = data_dir
        self.videos_classes = videos_classes
        self.classes_labels = classes_labels
        self.augmentation = augmentation
        self.frame_per_clip = frame_per_clip

        self.videos = np.asarray(list(videos_classes.keys()))

        if not isinstance(data_shape, tuple):
            data_shape = tuple(data_shape)
        self.data_shape = data_shape
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]

        self.data_name = data_name
        self.label_name = label_name

        #self.provide_data = [(data_name, (batch_size, frame_per_clip, ) + self.data_shape)]
        self.provide_data = [(data_name, (batch_size, ) + self.data_shape)]
        self.provide_label = [(label_name, (batch_size, ))]

        self.size = self.videos.shape[0]
        self.shuffle = shuffle
        self.cur = 0
        self.index = np.arange(self.size)

        self.reset()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            batch = self.get_batch()
            self.cur += self.batch_size
            return batch
        else:
            raise StopIteration

    def getindex(self):
        return self.cur/self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        #batch_data = nd.empty((self.batch_size, self.frame_per_clip,) + self.data_shape)
        batch_data = nd.empty((self.batch_size, ) + self.data_shape)
        batch_label = nd.empty(self.batch_size)
        batch_index = 0
        for i in xrange(self.cur, self.cur+self.batch_size):
            batch_data[batch_index][:] = self.read_one_video_random(self.videos[self.index[i]])
            batch_label[batch_index][:] = self.classes_labels[self.videos_classes[self.videos[self.index[i]]]]
            batch_index += 1

        return mx.io.DataBatch(data=[batch_data], label=[batch_label], pad=self.getpad(), index=self.getindex())

    def read_one_video_random(self, video_name):
        """
        Given the video, we divide it into K segments of equal duration, and then sample one snippet randomly from
        its corresponding segment
        :param video_name:
        :return:
        """
        video_path = os.path.join(self.data_dir[0], video_name, '')
        frames_name = [f for f in sorted(os.listdir(video_path)) if f.endswith('.jpg')]
        video_segments = np.linspace(start=0, stop=len(frames_name),
                                     num=self.frame_per_clip+1, endpoint=True, dtype=np.int16)
        video_clip_indices = map(lambda x: np.random.randint(low=video_segments[x], high=video_segments[x+1]),
                                 range(self.frame_per_clip))

        frames = []
        for i in xrange(self.frame_per_clip):
            frame_path = os.path.join(self.data_dir[0], video_name, frames_name[video_clip_indices[i]])
            frames.append(load_one_image(frame_path, greyscale=False))

        clip = mx.ndarray.concatenate(frames, axis=2)
        clip = pre_process_image(self.data_shape, clip, self.augmentation)
        clip = post_process_image(clip)

        #return mx.nd.reshape(clip, shape=(self.frame_per_clip, ) + self.data_shape)
        return clip

    def read_one_video(self, video_name):
        video_path = os.path.join(self.data_dir[0], video_name, '')
        logging.debug("The video path is {}".format(video_path))
        frames_name = [f for f in sorted(os.listdir(video_path)) if f.endswith('.jpg')]

        # based on the number
        time_step = int(len(frames_name)/self.frame_per_clip)
        start_frame_index = np.random.randint(len(frames_name) - (self.frame_per_clip - 1) * time_step)
        #start_frame_index = 0
        logging.debug("start frame index: {}; time_step: {}".format(start_frame_index, time_step))
        frames = []
        for _ in xrange(self.frame_per_clip):
            frame_path = os.path.join(self.data_dir[0], video_name, frames_name[start_frame_index])
            frames.append(load_one_image(frame_path, greyscale=False))
            start_frame_index += time_step

        clip = mx.ndarray.concatenate(frames, axis=2)
        clip = pre_process_image(self.data_shape, clip, self.augmentation)
        clip = post_process_image(clip)

        return mx.nd.reshape(clip, shape=(self.frame_per_clip, ) + self.data_shape)

    @property
    def data_size(self):
        return self.size
