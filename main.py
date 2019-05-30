from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import pickle
import argparse
import numpy as np
import tensorflow as tf

FLAGS = None


def main(_):
    tmp = pickle.load(open("data/cifar10_train.pkl", "rb"))
    x_train = tmp['data'].astype(np.float32).reshape([-1, 32, 32, 3]) / 127.5 - 1.
    
    model.fit(x_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
