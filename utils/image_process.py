import mxnet as mx
import mxnet.ndarray as nd
from augmentation import *


def load_one_image(img_path, record=None, lst_dict=None, ctx=None):
    if ctx is None:
        ctx = [mx.cpu()]

    if record is not None and lst_dict is not None:
        _, img = mx.recordio.unpack_img(record.read_idx(lst_dict[img_path]))
        return mx.nd.array(img, ctx=ctx)
    else:
        with open(img_path, 'rb') as fp:
            image_info = fp.read()

        return mx.img.imdecode(image_info, ctx=ctx)


def post_process_image(image):
    """
    Transform the image to make it shape as (channel, height, width)
    """
    return nd.transpose(image, axes=(2, 0, 1))


def pre_process_image(data_shape, image, augmentation):
    """
    Transforms input data with specified augmentation.
    """
    c, h, w = data_shape
    for process in augmentation:
        if process == 'random_crop':
            image = random_crop(image, w, h)
        elif process == 'random_horizon_flip':
            image = random_horizon_flip(image)
        elif process == 'random_corner_crop':
            image = random_corner_crop(image, w, h)
        elif process == 'left_top_corner_crop':
            image = left_top_corner_crop(image, w, h)
        elif process == 'left_bottom_corner_crop':
            image = left_bottom_corner_crop(image, w, h)
        elif process == 'right_top_corner_crop':
            image = right_top_corner_crop(image, w, h)
        elif process == 'right_bottom_corner_crop':
            image = right_bottom_corner_crop(image, w, h)
        elif process == 'centre_crop':
            image = centre_crop(image, w, h)
        elif process == 'random_border25_crop':
            image = random_border25_crop(image, w, h)
        elif process == 'horizon_flip':
            image = horizon_flip(image)
        else:
            raise NotImplementedError("This augmentation operation has not been implemented!")

    image_h, image_w, _ = image.shape
    if image_h != h or image_w != w:
        image = mx.img.imresize(image, w, h)

    return image