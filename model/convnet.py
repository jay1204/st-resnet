import mxnet as mx
from utils import load_pretrained_model, process_lst_file
from video_iter import VideoIter
from temporal_iter import TemporalIter
import numpy as np
import mxnet.ndarray as nd
from utils import load_one_image, post_process_image, pre_process_image
import os
import logging
import json
import time
from symbol import get_resnet


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
                                                                   ctx=self.ctx)
            #self.set_use_global_stats_json()
            #symbol = mx.symbol.load(self.model_params.dir + self.model_params.name + '-symbol.json')
            # adjust the network to satisfy the required input
            if self.mode == 'spatial':
                new_symbol, new_arg_params = self.refactor_model_spatial(symbol, arg_params)
                new_aux_params = aux_params
            elif self.mode == 'temporal':
                new_symbol, new_arg_params, new_aux_params = self.refactor_model_temporal(symbol, arg_params, aux_params)
            else:
                raise NotImplementedError('The refactoring method-{} for the model has not be implemented yet'.format(self.mode))

            new_symbol.save(self.model_params.dir + self.model_params.name + '-' + self.mode + '-symbol.json')
            self.set_use_global_stats_json()
            new_symbol = mx.symbol.load(self.model_params.dir +
                                        self.model_params.name + '-' + self.mode + '-symbol.json')
            return new_symbol, new_arg_params, new_aux_params

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
            data = mx.sym.Variable(name="data")
            symbol = get_resnet(data, drop_out=self.train_params.drop_out)
            all_layers = symbol.get_internals()
            net = all_layers['flatten0_output']
            net = mx.symbol.Dropout(net, p=self.train_params.drop_out, name='flatten0_dropout')
            #net = mx.symbol.FullyConnected(data=net, num_hidden=2048, name='fc2')
            #net = mx.symbol.Activation(name='relu_fc2', data=net, act_type='relu')
            #net = mx.symbol.Dropout(net, p=self.train_params.drop_out)
            net = mx.symbol.FullyConnected(data=net, num_hidden=self.num_classes, name='fc1')
            new_symbol = mx.symbol.SoftmaxOutput(data=net, name='softmax')
            new_arg_params = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
        elif self.model_params.name=='vgg16':
            all_layers = symbol.get_internals()
            net = all_layers['relu6_output']
            net = mx.symbol.Dropout(net, p = self.train_params.drop_out)
            net = mx.symbol.FullyConnected(data=net, num_hidden=4096, name='fc7')
            net = mx.symbol.Activation(name='relu7', data=net, act_type='relu')
            net = mx.symbol.Dropout(net, p=self.train_params.drop_out)
            net = mx.symbol.FullyConnected(data=net, num_hidden=self.num_classes, name='fc8')
            new_symbol = mx.symbol.SoftmaxOutput(data=net, name='softmax')
            new_arg_params = dict({k:arg_params[k] for k in arg_params if 'fc8' not in k})
        else:
            raise NotImplementedError('This model-{} has not been refactored!'.format(self.model_params.name))

        return new_symbol, new_arg_params

    def refactor_model_temporal(self, symbol, arg_params, aux_params):
        """
        Adjust the input to accomodate the flow frames
        :param symbol:
        :param arg_params:
        :return:
        """
        if self.model_params.name == 'resnet-50':
            data = mx.sym.Variable(name="data")
            symbol = get_resnet(data, drop_out=self.train_params.drop_out)
            all_layers = symbol.get_internals()
            net = all_layers['flatten0_output']
            net = mx.symbol.Dropout(net, p=self.train_params.drop_out, name='flatten0_dropout')
            # net = mx.symbol.FullyConnected(data=net, num_hidden=2048, name='fc2')
            # net = mx.symbol.Activation(name='relu_fc2', data=net, act_type='relu')
            # net = mx.symbol.Dropout(net, p=self.train_params.drop_out)
            net = mx.symbol.FullyConnected(data=net, num_hidden=self.num_classes, name='fc1')
            new_symbol = mx.symbol.SoftmaxOutput(data=net, name='softmax')
            new_arg_params = dict({k: arg_params[k] for k in arg_params if 'fc1' not in k})

            new_arg_params['bn_data_gamma'] = mx.ndarray.repeat(
                mx.ndarray.mean(new_arg_params['bn_data_gamma']),repeats=self.train_params.frame_per_clip * len(self.data_params.dir))
            new_arg_params['bn_data_beta'] = mx.ndarray.repeat(
                mx.ndarray.mean(new_arg_params['bn_data_beta']),repeats=self.train_params.frame_per_clip * len(self.data_params.dir))

            new_arg_params['conv0_weight'] = mx.ndarray.repeat(
                mx.ndarray.mean(new_arg_params['conv0_weight'], axis=1, keepdims=True),
                repeats=self.train_params.frame_per_clip * len(self.data_params.dir), axis=1)

            new_aux_params = dict({k: aux_params[k] for k in aux_params})

            new_aux_params['bn_data_moving_mean'] = mx.ndarray.repeat(
                mx.ndarray.mean(new_aux_params['bn_data_moving_mean']),
                repeats=self.train_params.frame_per_clip * len(self.data_params.dir))
            new_aux_params['bn_data_moving_var'] = mx.ndarray.repeat(
                mx.ndarray.mean(new_aux_params['bn_data_moving_var']),
                repeats=self.train_params.frame_per_clip * len(self.data_params.dir))

        else:
            raise NotImplementedError('This model-{} has not been refactored!'.format(self.model_params.name))

        return new_symbol, new_arg_params, new_aux_params

    def resume_training(self):
        return mx.model.load_checkpoint(self.model_params.dir + self.model_params.name + '-' + self.mode,
                                        self.train_params.load_epoch)

    def train(self):

        #record = None
        #lst_dict = None
        #if self.data_params.rec_file is not None and self.data_params.idx_file is not None \
        #        and self.data_params.lst_file is not None:
        #    record = mx.recordio.MXIndexedRecordIO(self.data_params.idx_file, self.data_params.rec_file, 'r')
        #    lst_dict = process_lst_file(self.data_params.lst_file)

        train_iter = self.create_train_iter(train=True)
        #train_iter2 = self.create_train_iter(train=True)
        #print train_iter1.provide_data, train_iter2.provide_data

        #train_iter = PrefetchingIter([train_iter1, train_iter2], rename_data = [{'data':'data_1'}, {'data':'data_2'}])
        #print train_iter.provide_data

        valid_iter = self.create_train_iter(train=False)

        net, arg_params, aux_params = self.configure_model()

        mod = mx.mod.Module(symbol=net, context=self.ctx) # , fixed_params_names=net.list_auxiliary_states())
        mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
        mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2))
        mod.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True)

        lr_sch = mx.lr_scheduler.MultiFactorScheduler(step=self.train_params.schedule_steps, factor=0.1)
        #adam = mx.optimizer.Optimizer.create_optimizer('adam', learning_rate=self.train_params.learning_rate,
         #                                              lr_scheduler=lr_sch,
         #                                              rescale_grad=1.0/self.train_params.batch_size)

        #mod.init_optimizer(optimizer=adam)
        #sgd = mx.optimizer.Optimizer.create_optimizer('sgd', learning_rate = self.train_params.learning_rate,
        #                                              momentum=0.9, wd=0.0005, lr_scheduler=lr_sch)
        mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', self.train_params.learning_rate),
                                                              ('momentum', 0.9), ('wd', 0.0001),
                                                              ('rescale_grad', 1.0/self.train_params.batch_size),
                                                              ('lr_scheduler', lr_sch)))
        #mod.init_optimizer(optimizer=sgd)
        #mod.init_optimizer(optimizer='adam', optimizer_params=(('learning_rate', self.train_params.learning_rate),
        #                                                       ('lr_scheduler', lr_sch),
        #                                                       ('epsilon', self.train_params.epsilon)))
        metric = mx.metric.create(['loss','acc'])
        count = 1
        train_acc = []
        valid_acc = []
        valid_accuracy = 0.6796

        train_iter.reset()
        metric.reset()

        logging.info("The dropout rate is {}, learning rate is {} and saving epoch is {}".
                    format(self.train_params.drop_out, self.train_params.learning_rate, self.train_params.load_epoch))
        for batch in train_iter:
            logging.debug('The time when I get the {}-th batch: {}'.format(count, time.asctime(time.localtime(time.time()))))
            mod.forward(batch, is_train=True)
            #print 'get the {}-th batch successfully'.format(count)
            mod.backward()
            #print mod.get_input_grads()[0].asnumpy()
            mod.update()
            mod.update_metric(metric, batch.label)
            #logging.info('The current iteration is %d'%(count))
            if count%100==0:
                #logger.info('Current optimizer parameters: ')
                #mod.forward(batch, is_train=False)
                #mod.update_metric(metric, batch.label)
                #print mod.get_outputs()
                train_acc.append(metric.get()[1][1])
                logging.info("The training loss of the %d-th iteration is %f, accuracy  is %f%%" %\
                      (count, metric.get()[1][0], metric.get()[1][1]*100))
                score = mod.score(valid_iter, ['loss','acc'], num_batch=20)
                valid_acc.append(score[1][1])
            #if count%100==0:
            #    va = self.evaluate(mod)
            #    valid_acc.append(va)
            #    print "The validation accuracy of the %d-th iteration is %f%%"%(count, valid_acc[-1] * 100)
                logging.info("The valid loss of the %d-th iteration is %f, accuracy is %f%%"%\
                     (count, score[0][1], score[1][1]*100))
                if valid_acc[-1] > valid_accuracy:
                    valid_accuracy = valid_acc[-1]
                    mod.save_checkpoint(self.model_params.dir + self.model_params.name + '-' + self.mode,
                                        self.train_params.load_epoch)
                # reset the metric for measuring the next 100 training batches
                metric.reset()

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
                print mod.get_outputs()[0].asnumpy()
                probs += mod.get_outputs()[0].asnumpy().sum(axis=0)

            pred_label = np.argmax(probs)
            acc += (pred_label == label)
            count += 1
            if count%1000==0:
                logging.info('Have finished processing the {}-th videos'.format(count))

        return acc/count

    def test_dataset_evaluation(self):
        #sym, args, auxs = mx.model.load_checkpoint(
        #    self.model_params.dir + self.model_params.name + '-' + self.mode, self.test_params.load_epoch)
        #if self.test_params.remove_softmax_layer:
        #    sym = all_layers['fc1_output']

        #mod = mx.module.Module(symbol=sym, context=self.ctx)
        #mod._arg_params = args
        #mod._aux_params = auxs
        #mod.set_params(arg_params=args, aux_params=auxs, allow_missing=True)
        #mod.params_initialized = True
        #mod = mx.module.Module.load(self.model_params.dir + self.model_params.name + '-' + self.mode,
        #                            self.test_params.load_epoch, context=self.ctx)

        net, arg_params, aux_params = mx.model.load_checkpoint(
            self.model_params.dir + self.model_params.name + '-' + self.mode, self.test_params.load_epoch)

        all_layers = net.get_internals()
        net = all_layers['fc1_output']
        mod = mx.mod.Module(symbol=net, context=self.ctx)
        mod.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True)

        test_accuracy = self.evaluate(mod)
        logging.info("The testing accuracy is %f%%" % (test_accuracy*100))

        return

    def freeze_mean_variance_batch_normalization_layers(self, symbol):
        pass

    def set_use_global_stats_json(self):
        json_file = json.loads(open(self.model_params.dir + self.model_params.name + '-' + self.mode + '-symbol.json').read())
        if self.train_params.use_global_stats:
            operator = "True"
            momentum = "1.0"
        else:
            operator = "False"
            momentum = "0.9"

        for param in json_file['nodes']:
            if param['op'] == 'BatchNorm' and self.mode == 'temporal' and param['name'] == 'bn_data':
                param['attr']['use_global_stats'] = "False"
                param['attr']['momentum'] = "0.9"
            elif param['op'] == 'BatchNorm':
                param['attr']['use_global_stats'] = operator
                param['attr']['momentum'] = momentum

        with open(self.model_params.dir + self.model_params.name + '-' + self.mode + '-symbol.json', 'w') as f:
            json.dump(json_file, f)
        return

    def create_train_iter(self, train=True, record=None, lst_dict=None):
        """
        Create training and validation data iterator

        :param train: bool, true means training data, false means validation data
        :param record:
        :param lst_dict:
        :return: VideoIter
        """
        if train:
            videos_classes = self.train_videos_classes
        else:
            videos_classes = self.test_videos_classes

        greyscale = False
        if self.mode == 'temporal':
            greyscale = True

        if train:
            return VideoIter(batch_size=self.train_params.batch_size, data_shape=self.data_params.data_shape,
                             data_dir=self.data_params.dir, videos_classes=videos_classes,
                             classes_labels=self.classes_labels, ctx=self.ctx, data_name='data',
                             label_name='softmax_label', mode='train', augmentation=self.train_params.augmentation,
                             frame_per_clip=self.train_params.frame_per_clip, greyscale=greyscale,
                             lst_dict=lst_dict, record=record,
                             multiple_processes=10)
        else:
            return VideoIter(batch_size=self.train_params.batch_size, data_shape=self.data_params.data_shape,
                             data_dir=self.data_params.dir, videos_classes=videos_classes,
                             classes_labels=self.classes_labels, ctx=self.ctx, data_name='data',
                             label_name='softmax_label', mode='train', augmentation=self.train_params.augmentation,
                             frame_per_clip=self.train_params.frame_per_clip, greyscale=greyscale,
                             lst_dict=lst_dict, record=record,
                             multiple_processes=3)




