import mxnet as mx
import random


def horizon_flip(src, p=0.5):
    if random.random() < p:
        src = nd.flip(src, axis=1)

    return src