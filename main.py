from __future__ import division, print_function, absolute_import
import os, sys, pickle, argparse
import numpy as np
import tensorflow as tf

from models import P3WGAN

FLAGS = None


def main(_):
    tmp = pickle.load(open('data/cifar10/cifar10_train.pkl', 'rb'))
    x_train = tmp['data'].astype(np.float32).reshape([-1, 32, 32, 3]) / 127.5 - 1.

    model = P3WGAN(gamma0=FLAGS.gamma0,
                   gamma1=FLAGS.gamma1,
                   gamma_steps=FLAGS.gamma_steps,
                   num_training_mover=FLAGS.num_training_mover,
                   num_training_generator=FLAGS.num_training_generator,
                   num_training_critic=FLAGS.num_training_critic,
                   num_mov_layers=FLAGS.num_mov_layers,
                   num_gen_feature_maps=FLAGS.num_gen_feature_maps,
                   num_cri_feature_maps=FLAGS.num_cri_feature_maps,
                   critic_atv=tf.nn.tanh,
                   num_z=FLAGS.num_z,
                   z_prior=z_prior,
                   batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   beta1=FLAGS.adam_beta1,
                   beta2=FLAGS.adam_beta2,
                   decay=FLAGS.learning_decay,
                   decay_steps=FLAGS.decay_steps,
                   num_epochs=FLAGS.num_epochs,
                   img_size=(32, 32, 3),
                   samples_fp='samples/samples_{epoch:04d}.png',
                   samples_h_fp='samples_h/samples_{epoch:04d}.png')

    model.fit(x_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma0', type=float, default=0.1,
                        help='Starting value of Gamma.')
    parser.add_argument('--gamma1', type=float, default=100.0,
                        help='Ending value of Gamma.')
    parser.add_argument('--gamma_steps', type=int, default=500,
                        help='Number of epochs until ending value of Gamma.')

    parser.add_argument('--num_training_mover', type=int, default=5,
                        help='Number of Mover updates per batch iteration.')
    parser.add_argument('--num_training_generator', type=int, default=1,
                        help='Number of Generator updates per batch iteration.')
    parser.add_argument('--num_training_critic', type=int, default=1,
                        help='Number of Critic updates per batch iteration.')

    parser.add_argument('--num_mov_layers', type=int, default=4,
                        help='Number of layers of Mover.')
    parser.add_argument('--num_gen_feature_maps', type=int, default=128,
                        help='Number of feature maps of Generator.')
    parser.add_argument('--num_cri_feature_maps', type=int, default=128,
                        help='Number of feature maps of Critic.')

    parser.add_argument('--num_z', type=int, default=128,
                        help='Number of latent units.')
    parser.add_argument('--z_prior', type=str, default='uniform',
                        help='Prior distribution of the noise (Uniform/Normal).')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')

    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Learning rate of 3 nets.')
    parser.add_argument('--adam_beta1', type=float, default=0.0,
                        help='Beta 1 in Adam optimiser.')
    parser.add_argument('--adam_beta2', type=float, default=0.9,
                        help='Beta 2 in Adam optimiser.')
    parser.add_argument('--learning_decay', type=int, default=1,
                        help='Boolean flag for learning rate decay.')
    parser.add_argument('--decay_steps', type=int, default=100000,
                        help='Decay steps for learning rate decay.')

    parser.add_argument('--num_epochs', type=int, default=500,
                        help='Number of epochs.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
