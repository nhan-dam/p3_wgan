from __future__ import division, print_function, absolute_import
import numpy as np


class Prior(object):
    def __init__(self, type):
        self.type = type

    def sample(self, shape):
        if self.type == 'uniform':
            return np.random.uniform(-1.0, 1.0, shape)
        else:
            return np.random.normal(0, 1, shape)


def make_batches(size, batch_size):
    '''
    Returns a list of batch indices (tuples of indices).
    '''
    return [(i, min(size, i + batch_size)) for i in range(0, size, batch_size)]


def create_image_grid(x, img_size, tile_shape):
    assert (x.shape[0] == tile_shape[0] * tile_shape[1])
    assert (x[0].shape == img_size)

    img = np.zeros((img_size[0] * tile_shape[0] + tile_shape[0] - 1,
                    img_size[1] * tile_shape[1] + tile_shape[1] - 1,
                    3))

    for t in range(x.shape[0]):
        i, j = t // tile_shape[1], t % tile_shape[1]
        img[i * img_size[0] + i: (i + 1) * img_size[0] + i, j * img_size[1] + j: (j + 1) * img_size[1] + j] = x[t]

    return img
