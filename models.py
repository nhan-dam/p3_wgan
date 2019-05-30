from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from functools import partial

import os
import numpy as np
import tensorflow as tf
from ops import lrelu, linear, conv2d, deconv2d
from utils import make_batches, Prior, conv_out_size_same, create_image_grid

batch_norm = partial(tf.contrib.layers.batch_norm,
                     decay=0.9,
                     updates_collections=None,
                     epsilon=1e-5,
                     scale=True)

