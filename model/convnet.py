import mxnet as mx
from utils import load_pretrained_model
from video_iter import VideoIter
import numpy as np
import mxnet.ndarray as nd
from utils import load_one_image, post_process_image, pre_process_image
import os
from logger import logger


class ConvNet(object):
    """
    This class takes a pre-trained model(e.g. resnet-50, resnet-101), and further tune it on our video image data
    """
    def __init__(self, model_params, data_params, train_params, test_params, train_videos_classes,
                 test_videos_classes, classes_labels, num_classes, ctx, mode='spatial'):
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
        self.mode = mode

    def configure_model(self):
        # load pre-trained model
        if self.train_params.resume:
            return self.resume_training()
        else:
            symbol, arg_params, aux_params = load_pretrained_model(self.model_params.url_prefix, self.model_params.name,
                                                                   self.model_params.model_epoch, self.model_params.dir,
                                                                   ctx = self.ctx)

            # adjust the network to satisfy the required input
            if self.mode == 'spatial':
                new_symbol, new_arg_params = self.refactor_model_spatial(symbol, arg_params)
            elif self.mode == 'temporal':
                new_symbol, new_arg_params = self.refactor_model_temporal(symbol, arg_params)
            else:
                raise NotImplementedError('The refactoring method-{} for the model has not be implemented yet'.format(self.mode))

        return new_symbol, new_arg_params, aux_params

    def refactor_model_spatial(self, symbol, arg_params):
        """
        Adjust the number of output classes

        :param symbol: the pretrained network symbol
        :param arg_params: the argument parameters of the pretrained model
        :return:
            - new_symbol
            - new_arg_params
        """
        if self.model_params.name=='resnet-50':
            all_layers = symbol.get_internals()
            net = all_layers['flatten0_output']
            net = mx.symbol.Dropout(net, p=self.train_params.drop_out)
            #net = mx.symbol.FullyConnected(data=net, num_hidden=2048, name='fc2')
            #net = mx.symbol.Activation(name='relu_fc2', data=net, act_type='relu')
            #net = mx.symbol.Dropout(net, p=self.train_params.drop_out)
            net = mx.symbol.FullyConnected(data=net, num_hidden=self.num_classes, name='fc1')
            new_symbol = mx.symbol.SoftmaxOutput(data=net, name='softmax')
            new_arg_params = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
        elif self.model_params.name=='vgg16':
            all_layers = symbol.get_internals()
            net = all_layers['relu6_output']
            net = mx.symbol.Dropout(net, p=self.train_params.drop_out)
            net = mx.symbol.FullyConnected(data=net, num_hidden=4096, name='fc7')
            net = mx.symbol.Activation(name='relu7', data=net, act_type='relu')
            net = mx.symbol.Dropout(net, p = self.train_params.drop_out)
            net = mx.symbol.FullyConnected(data=net, num_hidden=self.num_classes, name='fc8')
            new_symbol = mx.symbol.SoftmaxOutput(data=net, name='softmax')
            new_arg_params = dict({k:arg_params[k] for k in arg_params if 'fc8' not in k})
        else:
            raise NotImplementedError('This model has not been implemented!')

        return new_symbol, new_arg_params

    def refactor_model_temporal(self, symbol, arg_params):
        """
        Adjust the input to accomodate the flow frames
        :param symbol:
        :param arg_params:
        :return:
        """
        pass



    def resume_training(self):
        return mx.model.load_checkpoint(self.model_params.dir + self.model_params.name + '-' + self.mode,
                                        self.train_params.load_epoch)

    def train(self):
        net, arg_params, aux_params = self.configure_model()

        train_iter = VideoIter(batch_size=self.train_params.batch_size, data_shape=self.model_params.data_shape,
                               data_dir=self.data_params.dir, videos_classes=self.train_videos_classes,
                               classes_labels=self.classes_labels, ctx=self.ctx, data_name='data',
                               label_name='softmax_label', mode='train', augmentation=self.train_params.augmentation)
        valid_iter = VideoIter(batch_size=self.train_params.batch_size, data_shape=self.model_params.data_shape,
                               data_dir=self.data_params.dir, videos_classes=self.test_videos_classes,
                               classes_labels=self.classes_labels, ctx=self.ctx, data_name='data',
                               label_name='softmax_label', mode='train', augmentation=self.train_params.augmentation)

        mod = mx.mod.Module(symbol=net, context=self.ctx)
        mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
        mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2))
        mod.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True)

        lr_sch = mx.lr_scheduler.MultiFactorScheduler(step=[4000, 8000, 10000], factor=0.1)
        #mod.init_optimizer(optimizer='adam', optimizer_params=(('learning_rate', self.train_params.learning_rate),
        #                                                     ('lr_scheduler', lr_sch)))
        #sgd = mx.optimizer.Optimizer.create_optimizer('sgd', learning_rate = self.train_params.learning_rate,
        #                                              momentum=0.9, wd=0.0005, lr_scheduler=lr_sch)
        mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', self.train_params.learning_rate),
                                                              ('momentum', 0.9), ('wd',0.0005),
                                                              ('lr_scheduler', lr_sch)))
        #mod.init_optimizer(optimizer=sgd)


        metric = mx.metric.create(['loss','acc'])
        count = 1
        train_acc = []
        valid_acc = []
        valid_accuracy = 0.0

        train_iter.reset()
        metric.reset()

        logger.info("The dropout rate is {}, learning rate is {} and saving epoch is {}".
                    format(self.train_params.drop_out, self.train_params.learning_rate, self.train_params.load_epoch))
        for batch in train_iter:
            mod.forward(batch, is_train=True)
            mod.update_metric(metric, batch.label)
            mod.backward()
            mod.update()
            if count%100==0:
                logger.info('Current optimizer parameters: ')
                mod.forward(batch, is_train=False)
                mod.update_metric(metric, batch.label)
                train_acc.append(metric.get()[1][1])
                logger.info("The training loss of the %d-th iteration is %f, accuracy  is %f%%" %\
                      (count, metric.get()[1][0], metric.get()[1][1]*100))
                score = mod.score(valid_iter, ['loss','acc'], num_batch=20)
                valid_acc.append(score[1][1])
            #if count%100==0:
            #    va = self.evaluate(mod)
            #    valid_acc.append(va)
            #    print "The validation accuracy of the %d-th iteration is %f%%"%(count, valid_acc[-1] * 100)
                logger.info("The valid loss of the %d-th iteration is %f, accuracy is %f%%"%\
                     (count, score[0][1], score[1][1]*100))
                if valid_acc[-1] > valid_accuracy:
                    valid_accuracy = valid_acc[-1]
                    mod.save_checkpoint(self.model_params.dir + self.model_params.name + '-' + self.mode,
                                        self.train_params.load_epoch)
            count += 1
            if count > self.train_params.iteration:
                break

        return train_acc, valid_acc

    def evaluate(self, mod):
        acc = 0.0
        count = 0
        for video, video_class in self.test_videos_classes.items():
            label = 0
            probs = np.zeros(self.num_classes)
            for aug in self.test_params.augmentation:
                valid_iter = VideoIter(batch_size=self.test_params.clip_per_video,
                                       data_shape=self.model_params.data_shape,
                                       data_dir=self.data_params.dir, videos_classes={video: video_class},
                                       classes_labels=self.classes_labels, ctx=self.ctx, data_name='data',
                                       label_name='softmax_label', mode='test',
                                       augmentation=aug, clip_per_video=self.test_params.clip_per_video)
                if not mod.binded:
                    mod.bind(data_shapes=valid_iter.provide_data, label_shapes=valid_iter.provide_label)
                batch = valid_iter.next()
                label = batch.label[0].asnumpy().astype(int)[0]
                mod.forward(batch, is_train=False)
                probs += mod.get_outputs()[0].asnumpy().sum(axis=0)

            pred_label = np.argmax(probs)
            acc += (pred_label == label)
            count += 1

        return acc/count

    def test_dataset_evaluation(self):
        mod = mx.module.Module.load(self.model_params.dir + self.model_params.name, self.test_params.load_epoch,
                                    context=self.ctx)

        test_accuracy = self.evaluate(mod)
        logger.info("The testing accuracy is %f%%" % (test_accuracy*100))











