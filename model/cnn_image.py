import mxnet as mx
from utils import load_pretrained_model
from video_iter import VideoIter
import numpy as np
import mxnet.ndarray as nd
from utils import load_one_image, post_process_image, pre_process_image
import os


class CNN_Image(object):
    """
    This class takes a pre-trained model(e.g. resnet-50, resnet-101), and further tune it on our video image data
    """
    def __init__(self, model_params, data_params, train_params, test_params, train_videos_classes,
                 test_videos_classes, classes_labels, num_classes, ctx):
        """

        :param model_params: a dict of the pre-trained network setting, including the model name, directory, etc.
        :param data_params: a dict of images info, including the directory of images, image size
        :param train_params: a dict of parameters used in training
        :param ctx: the device context(Context or list of Context)
        """
        self.model_params = model_params
        self.data_params = data_params
        self.train_params = train_params
        self.test_params = test_params
        self.ctx = ctx
        self.train_videos_classes = train_videos_classes
        self.test_videos_classes = test_videos_classes
        self.classes_labels = classes_labels
        self.num_classes = num_classes

    def configure_model(self):
        # load pre-trained model
        symbol, arg_params, _ = load_pretrained_model(self.model_params.url_prefix,
                                                   self.model_params.name,
                                                   self.model_params.model_epoch,
                                                   self.model_params.dir,
                                                   ctx = self.ctx)

        # adjust the network to satisfy the required input
        new_symbol, new_arg_params = self.refactor_model(symbol, arg_params)
        return new_symbol, new_arg_params

    def refactor_model(self, symbol, arg_params):
        """
        Adjust the number of output classes

        :param symbol: the pretrained network symbol
        :param arg_params: the argument parameters of the pretrained model
        :return:
            - new_symbol
            - new_arg_params
        """
        all_layers = symbol.get_internals()
        net = all_layers['flatten0_output']
        net = mx.symbol.Dropout(net, p=self.train_params.drop_out)
        net = mx.symbol.FullyConnected(data=net, num_hidden=self.num_classes, name='fc1')
        new_symbol = mx.symbol.SoftmaxOutput(data=net, name='softmax')
        new_arg_params = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})

        return new_symbol, new_arg_params

    def train(self):
        net, args = self.configure_model()

        train_iter = VideoIter(batch_size=self.train_params.batch_size, data_shape=self.model_params.data_shape,
                               data_dir=self.data_params.dir, videos_classes=self.train_videos_classes,
                               classes_labels=self.classes_labels, ctx=self.ctx, data_name='data',
                               label_name='softmax_label', mode='train', augmentation=self.train_params.augmentation)

        valid_iter = VideoIter(batch_size=self.train_params.batch_size, data_shape=self.model_params.data_shape,
                               data_dir=self.data_params.dir, videos_classes=self.test_videos_classes,
                               classes_labels=self.classes_labels, ctx=self.ctx, data_name='data',
                               label_name='softmax_label', mode='train', augmentation=self.train_params.augmentation)
                               #frame_per_video=self.test_params.frame_per_video)

        mod = mx.mod.Module(symbol=net, context=self.ctx)
        mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
        mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2))

        lr_sch = mx.lr_scheduler.FactorScheduler(step=20000, factor=0.1)
        mod.init_optimizer(optimizer='adam', optimizer_params=(('learning_rate', self.train_params.learning_rate),
                                                              ('lr_scheduler', lr_sch)))

        metric = mx.metric.create(['loss','acc'])
        count = 1
        train_acc = []
        valid_acc = []
        valid_accuracy = 0.0
        for epoch in range(1, self.train_params.epoch + 1):
            train_iter.reset()
            metric.reset()
            for batch in train_iter:
                mod.forward(batch, is_train=True)
                mod.update_metric(metric, batch.label)
                mod.backward()
                mod.update()
                if count%1==0:
                    mod.forward(batch, is_train=False)
                    mod.update_metric(metric, batch.label)
                    train_acc.append(metric.get()[1][1])
                    print "The training loss of the %d-th iteration is %f, accuracy  is %f%%" %\
                          (count, metric.get()[1][0], metric.get()[1][1]*100)
                    #score = mod.score(valid_iter, ['loss','acc'], num_batch=10)
                    #valid_acc.append(score[1][1])
                    #print "The valid loss of the %d-th iteration is %f, accuracy is %f%%"%\
                    #      (count, score[0][1], score[1][1]*100)
                    #if valid_acc[-1] > valid_accuracy:
                    #    valid_accuracy = valid_acc[-1]
                    #    mod.save_checkpoint(self.model_params.dir + self.model_params.name, epoch)
                    self.evaluate(self.test_videos_classes, mod)
                count += 1
        return train_acc, valid_acc

    def evaluate(self, test_videos_classes, mod):
        acc = []
        c, h, w = self.model_params
        for video, video_class in test_videos_classes.items():
            video_path = os.path.join(self.data_params.dir, video, '')
            batch_data = nd.empty((self.test_params.frame_per_video, c, h, w))
            frames = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
            sample_gap = float(len(frames) - 1) / self.test_params.frame_per_video
            sample_frame_names = []
            for i in xrange(self.test_params.frame_per_video):
                sample_frame_names.append(frames[int(round(i * sample_gap))])

            for aug in self.test_params.augmentation:
                for j, sample_frame_name in enumerate(sample_frame_names):
                    sample_frame = self.read_image(os.path.join(video_path, sample_frame_name), aug)
                    batch_data[j][:] = sample_frame

                mod.forward(batch_data, is_train=False)
                result = mod.get_outputs()[0].asnumpy()
                print result

        return 0

    @staticmethod
    def read_image(img_path, augmentation):
        image = load_one_image(img_path)
        image = pre_process_image(image, augmentation)
        image = post_process_image(image)

        return image









