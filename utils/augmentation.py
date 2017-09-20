import mxnet.ndarray as nd
import random
import mxnet as mx
import numpy as np


def random_crop(image, w, h):
    image, _ = mx.img.random_crop(image, (w, h))
    return image


def random_horizon_flip(image, p=0.5):
    if random.random() <= p:
        image = nd.flip(image, axis=1)

    return image


def horizon_flip(image):
    return nd.flip(image, axis=1)


def left_top_corner_crop(image, w, h):
    """

    :param image: mx.ndarray of shape(h,w,c)
    :param w: int crop width
    :param h: int crop height
    :return:
    """
    return mx.img.fixed_crop(image, x0=0, y0=0, w=w, h=h)


def left_bottom_corner_crop(image, w, h):
    """

    :param image:
    :param w:
    :param h:
    :return:
    """
    image_h, image_w, _ = image.shape
    return mx.img.fixed_crop(image, x0=0, y0=image_h-h, w=w, h=h)


def right_top_corner_crop(image, w, h):
    """

    :param image: mx.ndarray of shape(h,w,c)
    :param w: int crop width
    :param h: int crop height
    :return:
    """
    _, image_w, _ = image.shape
    return mx.img.fixed_crop(image, x0=image_w-w, y0=0, w=w, h=h)


def right_bottom_corner_crop(image, w, h):
    """

    :param image: mx.ndarray of shape(h,w,c)
    :param w: int crop width
    :param h: int crop height
    :return:
    """
    image_h, image_w, _ = image.shape
    return mx.img.fixed_crop(image, x0=image_w-w, y0=image_h-h, w=w, h=h)


def centre_crop(image, w, h):
    """

    :param image: mx.ndarray of shape(h,w,c)
    :param w: int crop width
    :param h: int crop height
    :return:
    """
    image, _ = mx.img.center_crop(image, (w, h))
    return image


def random_corner_crop(image, w, h, w_range=(224, 257), h_range=(168, 193)):
    """
    First, crop the image based on the random sampling of the width from w_range, and height from h_range. Then randomly
    cropping from the borders and centre of the image. Finally, resize the cropped image to the required size of (w, h)

    :param image: mx.ndarray of shape(h,w,c)
    :param w: the width of the cropped image
    :param h: the height of the cropped image
    :param w_range: tuple of two elements, presenting the boundaries of the sample range for width
    :param h_range: tuple of two elements, presenting the boundaries of the sample range for height
    :return:
    """
    rw = np.random.randint(low=w_range[0], high=w_range[1])
    rh = np.random.randint(low=h_range[0], high=h_range[1])
    which_corner = np.random.randint(5)
    if which_corner == 0:
        image = left_top_corner_crop(image, w=rw, h=rh)
    elif which_corner == 1:
        image = left_bottom_corner_crop(image, w=rw, h=rh)
    elif which_corner == 2:
        image = right_top_corner_crop(image, w=rw, h=rh)
    elif which_corner == 3:
        image = mx.img.fixed_crop(image, w=rw, h=rh)
    else:
        image = centre_crop(image, rw, rh)

    return mx.img.imresize(image, w, h)


def random_border25_crop(image, w, h):
    """
    First, randomly jitter the width and height of the image by 25%, and also random crop it from a maximum of 25%
    distance from the image borders. Then, resize the image to w*h.

    :param image: mx.ndarray of shape(h,w,c)
    :param w: int crop width
    :param h: int crop height
    :return:
    """
    image_h, image_w, _ = image.shape
    rh = min(int(h * (0.75 + 0.5 * random.random())), image_h)
    rw = min(int(w * (0.75 + 0.5 * random.random())), image_w)

    h_border = np.random.randint(2)
    if h_border:  # has a maximum distance of 25% to the top or bottom borders
        top_border = np.random.randint(2)
        if top_border:
            y0 = np.random.randint(min(int(image_h * 0.25), image_h - rh))
        else:
            y0 = np.random.randint(max(int(image_h * 0.75), rh), image_h) - rh
        x0 = np.random.randint(image_w - rw)
    else:  # has a maximum distance of 25% to the left or right borders
        left_border = np.random.randint(2)
        if left_border:
            x0 = np.random.randint(min(int(image_w * 0.25), image_w - rw))
        else:
            x0 = np.random.randint(max(int(image_w * 0.75), rw), image_w) - rw
        y0 = np.random.randint(image_h - rh)

    image = mx.img.fixed_crop(image, x0=x0, y0=y0, w=rw, h=rh)
    return mx.img.imresize(image, w, h)






