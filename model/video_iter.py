import numpy as np
import os
import random
import mxnet as mx
from mxnet.executor_manager import _split_input_slice
import mxnet.ndarray as nd
from utils import load_one_image, post_process_image, pre_process_image
import multiprocessing, threading
import logging
import time


class VideoIter(mx.io.DataIter):
    """
    This class is a wrapper of the basic mx.io.DataIter.
    """
    def __init__(self, batch_size, data_shape, data_dir, videos_classes, classes_labels, ctx=None, data_name='data',
                 label_name='label', mode='train', augmentation=None, clip_per_video=1, frame_per_clip=1, lst_dict=None,
                 record=None, multiple_processes=2, multiple_threads=32):
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

        if batch_size%clip_per_video != 0:
            raise ValueError('The batch size is not an integral multiple of (frames per video)')

        self.batch_size = batch_size
        self.mode = mode

        if not isinstance(data_dir, list) or len(data_dir) <= 0:
            raise SyntaxError('The data_dir should be a list and contains at least some data directory')
        self.data_dir = data_dir
        self.videos_classes = videos_classes
        self.classes_labels = classes_labels
        self.augmentation = augmentation
        self.clip_per_video = clip_per_video
        self.frame_per_clip = frame_per_clip

        self.videos = np.asarray(list(videos_classes.keys()))

        if not isinstance(data_shape, tuple):
            data_shape = tuple(data_shape)
        c, h, w = data_shape
        self.data_shape = (c * len(self.data_dir) * self.frame_per_clip, h, w)
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]

        self.data_name = data_name
        self.label_name = label_name

        self.provide_data = [(data_name, (batch_size, ) + self.data_shape)]
        self.provide_label = [(label_name, (batch_size, ))]

        self.video_size = self.videos.shape[0]
        self.batch_videos = batch_size / clip_per_video

        self.cur = 0
        if record is not None and lst_dict is not None:
            self.record = record
            self.lst_dict = lst_dict
        else:
            self.record = None
            self.lst_dict = None

        self.reset()
        self.multiple_threads = multiple_threads
        # create a queue object
        self.q = multiprocessing.Queue(maxsize=2)
        self.pws = [multiprocessing.Process(target=self.write) for _ in range(multiple_processes)]
        for pw in self.pws:
            pw.daemon = True
            pw.start()

    def write(self):
        while True:
            self.q.put(obj=self.get_batch(), block=True, timeout=None)

    def reset(self):
        self.cur = 0

    def iter_next(self):
        if self.mode == 'train':
            return True
        else:
            return self.cur + self.batch_videos <= self.video_size

    def next(self):
        if self.q.empty():
            logging.debug("waiting for data")
        if self.iter_next():
            self.cur += self.batch_videos
            return self.q.get(block=True, timeout=None)
        #if self.iter_next():
        #    self.cur += self.batch_videos
        #    return self.get_batch()
        else:
            raise StopIteration

    def get_batch(self):
        np.random.seed()
        if self.mode == 'train':
            video_indices = np.random.choice(self.videos.shape[0], size=self.batch_size, replace=False)
            sample_videos = self.videos[video_indices]
            c, h, w = self.data_shape
            batch_data = nd.empty((self.batch_size, c, h, w))
            batch_label = nd.empty(self.batch_size)
            split_sample_indices = np.array_split(np.arange(self.batch_size), self.multiple_threads)
            prefetch_threads = []
            for sub_sample_indices in split_sample_indices:
                t = threading.Thread(target=self.read_train_frames,
                                     args=[batch_data, batch_label, sample_videos, sub_sample_indices])
                prefetch_threads.append(t)
                t.setDaemon(True)
                t.start()
            for t in prefetch_threads:
                t.join()
        else:
            sample_videos = self.videos[self.cur:(self.cur+self.batch_videos)]
            batch_data, batch_label = self.read_test_frames(sample_videos)

        return mx.io.DataBatch(data=[batch_data], label=[batch_label], pad=self.getpad(), index=self.getindex())

    def read_train_frames(self, batch_data, batch_label, sample_videos, sub_sample_indices):
        """
        Read a series of frames by sampling one frame from each video.

        :param sample_videos: numpy array of video name
        :return:
        """
        logging.debug("The start of current thread {}".format(time.asctime(time.localtime(time.time()))))
        for i in sub_sample_indices:
            video_path = os.path.join(self.data_dir[0], sample_videos[i], '')
            frames_name = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
            start_frame_index = np.random.randint(len(frames_name) - self.frame_per_clip)
            sample_clip = self.next_clip(sample_videos[i], frames_name, start_frame_index)
            batch_data[i][:] = sample_clip
            batch_label[i][:] = self.classes_labels[self.videos_classes[sample_videos[i]]]
        logging.debug("The end of current thread {}".format(time.asctime(time.localtime(time.time()))))

        return

    def read_test_frames(self, sample_videos):
        """
        Gather a set of test frames by uniformly sampling from each sample videos

        :param sample_videos:
        :return:
        """
        c, h, w = self.data_shape
        batch_data = nd.empty((self.batch_size, c, h, w))
        batch_label = nd.empty(self.batch_size)
        for i in xrange(sample_videos.shape[0]):
            video_path = os.path.join(self.data_dir[0], sample_videos[i], '')
            frames = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
            sample_gap = float(len(frames) - 1) / self.clip_per_video
            sample_frame_names = []
            for k in xrange(self.clip_per_video):
                sample_frame_names.append(frames[int(round(k * sample_gap))])

            for j, sample_frame_name in enumerate(sample_frame_names):
                sample_frame = self.next_clip(os.path.join(video_path, sample_frame_name))
                batch_data[i * self.clip_per_video + j][:] = sample_frame
                batch_label[i * self.clip_per_video + j][:] = self.classes_labels[self.videos_classes[sample_videos[i]]]

        return batch_data, batch_label

    def getpad(self):
        if self.cur + self.batch_videos > self.video_size:
            return (self.cur + self.batch_videos - self.video_size) * self.clip_per_video
        else:
            return 0

    def getindex(self):
        return self.cur/self.batch_videos

    def next_clip(self, video_name, frames_name, start_frame_index):
        c, h, w = self.data_shape
        sub_c = c / (len(self.data_dir) * self.frame_per_clip)
        clip = nd.empty((c, h, w))
        j = 0
        for dir in self.data_dir:
            video_path = os.path.join(dir, video_name, '')
            for i in xrange(start_frame_index, start_frame_index+self.frame_per_clip):
                frame_path = os.path.join(video_path, frames_name[i])
                clip[j:(j + sub_c)][:] = load_one_image(frame_path, record = self.record, lst_dict=self.lst_dict)
                j += sub_c

        #clip = mx.ndarray.concatenate(frames, axis=2)
        clip = pre_process_image(self.data_shape, clip, self.augmentation)
        clip = post_process_image(clip)

        return clip

