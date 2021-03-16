import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K

def read_img(img_path, preprocess_imput, size):
    img = cv2.imread(img_path)
    pimg = cv2.resize(img, size)

    pimg = np.expand_dims(pimg, axis=0)
    pimg = preprocess_input(pimg)

    return img, pimg

def deprocess_image(x):
    """util function to convert a tensor into a valid image.
    Args:
           x: tensor of filter.
    Returns:
           x: deprocessed tensor.
    """
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')

    return x

def normalize(x):
    """utility function to normalize a tensor by its L2 norm
    Args:
           x: gradient.
    Returns:
           x: gradient.
    """
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

    