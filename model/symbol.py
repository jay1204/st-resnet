import mxnet as mx
from mxnet.gluon.contrib.rnn import Conv2DLSTMCell
import logging
from mxnet.rnn import LSTMCell, BidirectionalCell

eps = 2e-5
use_global_stats = True
workspace = 512
res_deps = {'50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}
units = res_deps['50']
filter_list = [256, 512, 1024, 2048]
momentum = 1


def residual_unit(data, num_filter, stride, dim_match, name, drop_out):
    bn1   = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn1',
                             momentum=momentum)
    act1  = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2   = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn2',
                             momentum=momentum)
    act2  = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride, pad=(1, 1),
                               no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3   = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn3',
                             momentum=momentum)
    act3  = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
    #drop3 = mx.sym.Dropout(act3, p=drop_out, name=name + '_relu3' + '_dropout')
    conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                      workspace=workspace, name=name + '_sc')
    sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')
    return sum


def get_resnet(data, drop_out):
    # res1
    data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='bn_data')
    conv0   = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                               no_bias=True, name="conv0", workspace=workspace)
    bn0   = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn0',
                             momentum=momentum)
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    # res2
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1',
                         drop_out=drop_out)
    for i in range(2, units[0] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True,
                             name='stage1_unit%s' % i, drop_out=drop_out)
    conv_C2 = unit

    # res3
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1',
                         drop_out=drop_out)
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True,
                             name='stage2_unit%s' % i, drop_out=drop_out)
    conv_C3 = unit

    # res4
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1',
                         drop_out=drop_out)
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True,
                             name='stage3_unit%s' % i, drop_out=drop_out)
    conv_C4 = unit

    # res5
    unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1',
                         drop_out=drop_out)
    for i in range(2, units[3] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True,
                             name='stage4_unit%s' % i, drop_out=drop_out)

    # 1 * 1 kernel to reduce the number of feature maps to 256
    #unit = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='stage5_bn1')
    #unit = mx.sym.Activation(data=unit, act_type='relu', name='stage5_relu1')
    #unit = mx.sym.Convolution(data=unit, num_filter=256, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
    #                           no_bias=True, workspace=workspace, name='stage5_conv1')

    unit = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1',
                            momentum=momentum)
    unit = mx.sym.Activation(data=unit, act_type='relu', name='relu1')
    unit = mx.sym.Pooling(data=unit, global_pool=True, kernel=(7, 7), pool_type='avg', pad=(0,0), stride=(1,1), name='pool1')
    unit = mx.sym.Flatten(data=unit, name='flatten0')

    return unit


