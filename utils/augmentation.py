import mxnet.ndarray as nd
import random


def horizon_flip(src, p=0.5):
    if random.random() < p:
        src = nd.flip(src, axis=1)

    return src