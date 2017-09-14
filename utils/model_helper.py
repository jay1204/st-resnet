import numpy as np
import urllib
import os
import mxnet as mx
import logging


def get_model(prefix, code, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    download(prefix + '-symbol.json', model_dir)
    download(prefix + '-%04d.params' % code, model_dir)


# obtain the pre-trained model
def download(url, model_dir):
    filename = url.split('/')[-1]
    if not os.path.exists(model_dir + filename):
        urllib.urlretrieve(url, model_dir + filename)


def spec_context(param, ctx):
    """
    This func specifies the device context(computation source:CPU/GPU)
    of the NDArray

    Inputs:
        - param: dict of str to NDArray
        - ctx: the device context(Context or list of Context)

    Returns:
        None
    """
    for k, v in param.items():
        param[k] = v.as_in_context(ctx)

    return


def load_pretrained_model(prefix, model_name, epoch, model_dir, ctx=None):
    """
    This func is a wrapper of the mx.model.load_checkpoint. It can
    also specify the context(computation source:CPU/GPU) that will
    the params

    Inputs:
        - prefix: string indicating prefix of model name
        - epoch: int indicating epoch number
        - ctx: the device context(Context or list of Context)

    Returns:
        - arg_params: dict of str to NDArray of net's weights
        - aux_params: dict of str to NDArray of net's auxiliary states
    """
    get_model(prefix, epoch, model_dir)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_dir + model_name, epoch)
    logging.info('The pretrained model has been loaded successfully!')
    return sym, arg_params, aux_params
