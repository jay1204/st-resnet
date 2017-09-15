import mxnet as mx
from utils import load_pretrained_model
from image_iter import ImageIter


class ConvImage(object):
    """
    This class takes a pre-trained model(e.g. resnet-50, resnet-101), and further tune it on our video image data
    """
    def __init__(self, model_params, data_params, train_params, train_videos_classes,
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
        net = mx.symbol.FullyConnected(data=net, num_hidden=2048, name='fc1')
        net = mx.symbol.Activation(data=net, name="ReLU1", act_type="relu")
        net = mx.symbol.Dropout(net, p=self.train_params.dropout)
        net = mx.symbol.FullyConnected(data=net, num_hidden=self.num_classes, name='fc2')
        new_symbol = mx.symbol.SoftmaxOutput(data=net, name='softmax')
        new_arg_params = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})

        return new_symbol, new_arg_params

    def train(self):
        net, args = self.configure_model()

        train_iter = ImageIter(batch_size=self.train_params.batch_size, data_shape=self.model_params.data_shape,
                               data_dir=self.data_params.dir, videos_classes= self.train_videos_classes,
                               classes_labels=self.classes_labels,ctx=self.ctx, data_name='data',
                               label_name='softmax_label', mode='train', augmentation=self.train_params.augmentation)
        valid_iter = ImageIter(batch_size=self.train_params.batch_size, data_shape=self.model_params.data_shape,
                               data_dir=self.data_params.dir, videos_classes= self.test_videos_classes,
                               classes_labels=self.classes_labels,ctx=self.ctx, data_name='data',
                               label_name='softmax_label', mode='train', augmentation=self.train_params.augmentation)

        mod = mx.mod.Module(symbol=net, context=self.ctx)
        mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
        mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2))

        lr_sch = mx.lr_scheduler.FactorScheduler(step=1000, factor=0.5)
        mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', self.train_params.learning_rate),
                                                              ('lr_scheduler', lr_sch)))
        metric = mx.metric.create('acc')
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
                if count%100==0:
                    train_acc.append(metric.get()[1])
                    print "The training accuracy of the %d-th iteration is %f"%(count, train_acc[-1])
                    score = mod.score(valid_iter, ['acc'], num_batch=1)
                    valid_acc.append(score[0][1])
                    print "The valid accuracy of the %d-th iteration is %f"%(count, valid_acc[-1])
                    if valid_acc[-1] > valid_accuracy:
                        valid_accuracy = valid_acc[-1]
                        mod.save_checkpoint(self.model_params.dir + self.model_params.name, epoch)
                count += 1
        return train_acc, valid_acc




