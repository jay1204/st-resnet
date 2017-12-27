import mxnet as mx
import mxnet.ndarray as nd
from augmentation import *
import numpy


def load_one_image(img_path, greyscale = False, record=None, lst_dict=None):
    if record is not None and lst_dict is not None:
        _, img = mx.recordio.unpack_img(record.read_idx(lst_dict[img_path]))
        return mx.nd.array(img)
    else:
        with open(img_path, 'rb') as fp:
            image_info = fp.read()
        if greyscale:
            flag = 0
        else:
            flag = 1
        return mx.img.imdecode(image_info, flag=flag)


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
    if 'horizon_flip' in augmentation:
        image = horizon_flip(image)

    random_crop_list = ['random_crop', 'random_border25_crop', 'random_corner_crop']

    aug_random_crop_list = []
    for aug in augmentation:
        if aug in random_crop_list:
            aug_random_crop_list.append(aug)

    if len(aug_random_crop_list) > 0:
        selected = np.random.randint(len(aug_random_crop_list))
        if aug_random_crop_list[selected] == 'random_crop':
            image = random_crop(image, w, h)
        elif aug_random_crop_list[selected] == 'random_border25_crop':
            image = random_border25_crop(image, w, h)
        elif aug_random_crop_list[selected] == 'random_corner_crop':
            image = random_corner_crop(image, w, h)

    image_h, image_w, _ = image.shape
    if image_h != h or image_w != w:
        image = mx.img.imresize(image, w, h)

    return image