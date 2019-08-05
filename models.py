from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import os
from functools import partial
import scipy.misc as spm

from utils import make_batches, create_image_grid
from ops import linear, residual_block
from utils import Prior

batch_norm = partial(tf.contrib.layers.batch_norm, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)


class P3WGAN(object):
    '''
    Three-Player Wasserstein Generative Adversarial Network
    '''

    def __init__(self,
                 gamma0=1.0,
                 gamma1=10.0,
                 gamma_steps=50,
                 num_training_mover=5,
                 num_training_generator=1,
                 num_training_critic=1,
                 num_mov_layers=4,  # number of layers of Mover
                 num_gen_feature_maps=128,  # number of feature maps of Generator
                 num_cri_feature_maps=128,  # number of feature maps of Critic
                 critic_atv=None,
                 num_z=128,
                 z_prior='uniform',
                 batch_size=64,
                 learning_rate=0.0001,
                 beta1=0.0,  # \beta1 in Adam optimiser
                 beta2=0.9,  # \beta2 in Adam optimiser
                 decay=False,
                 decay_steps=100000,
                 img_size=(32, 32, 3),  # (height, width, channels)
                 num_iter_per_epoch=100032,
                 samples_fp=None,
                 samples_h_fp=None,
                 num_epochs=500,
                 random_seed=6789):
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.gamma_steps = gamma_steps
        self.num_training_mover = num_training_mover
        self.num_training_generator = num_training_generator
        self.num_training_critic = num_training_critic
        self.num_mov_layers = num_mov_layers
        self.num_gen_feature_maps = num_gen_feature_maps
        self.num_cri_feature_maps = num_cri_feature_maps
        self.critic_atv = critic_atv
        self.num_z = num_z
        self.z_prior = Prior(z_prior)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay = decay
        self.decay_steps = decay_steps
        self.img_size = img_size
        self.num_iter_per_epoch = num_iter_per_epoch
        self.samples_fp = samples_fp
        self.samples_h_fp = samples_h_fp
        self.num_epochs = num_epochs
        self.random_seed = random_seed

    def _init(self):
        self.epoch = 0
        self.tf_graph = tf.Graph()
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        self.tf_config.log_device_placement = False
        self.tf_config.allow_soft_placement = True
        self.tf_session = tf.Session(config=self.tf_config, graph=self.tf_graph)
        np.random.seed(self.random_seed)
        with self.tf_graph.as_default():
            tf.set_random_seed(self.random_seed)

    def _build_model(self):
        self.lr = tf.placeholder(tf.float32, name="learning_rate")
        self.gamma = tf.placeholder(tf.float32, name="gamma")
        self.x = tf.placeholder(tf.float32, [None, self.img_size[0], self.img_size[1], self.img_size[2]],
                                name="real_data")
        self.z = tf.placeholder(tf.float32, [None, self.num_z], name='noise')

        # create Generator G and sampler
        self.g = self._create_generator(self.z, name="generator")

        # create Mover H
        self.h = self._create_mover(self.x, name="mover")

        # create Critic F
        f_g = self._create_critic(self.g, name="critic")
        f_h = self._create_critic(self.h, name="critic", reuse=True)

        # define loss function
        self.efg = tf.reduce_mean(f_g)
        self.efh = tf.reduce_mean(f_h)
        mse = tf.reduce_mean(tf.square(self.x - self.h), axis=[1, 2, 3])
        sqrt_mse = tf.sqrt(mse + 1e-8)
        self.transport_cost = tf.reduce_mean(sqrt_mse)

        self.m_loss = self.gamma * self.transport_cost - self.efh
        self.c_loss = self.efh - self.efg
        self.g_loss = self.efg

        # create optimisers
        self.c_opt = self._create_optimizer(self.c_loss, lr=self.lr, bt1=self.beta1, bt2=self.beta2, scope="critic")
        self.g_opt = self._create_optimizer(self.g_loss, lr=self.lr, bt1=self.beta1, bt2=self.beta2, scope="generator")
        self.m_opt = self._create_optimizer(self.m_loss, lr=self.lr, bt1=self.beta1, bt2=self.beta2, scope="mover")

    def fit(self, x):
        if (not hasattr(self, 'epoch')) or self.epoch == 0:
            self._init()
            with self.tf_graph.as_default():
                self._build_model()
                self.tf_session.run(tf.global_variables_initializer())

        batch_idx_step = max(self.num_training_mover, self.num_training_generator, self.num_training_critic)
        num_data = x.shape[0] - x.shape[0] % (self.batch_size * batch_idx_step)
        if num_data > 150000:
            num_data = self.num_iter_per_epoch

        batches = make_batches(num_data, self.batch_size * batch_idx_step)

        iter = 0.0
        while (self.epoch < self.num_epochs):
            for batch_idx in np.arange(0, len(batches), batch_idx_step):
                iter += 1.0
                if self.decay:
                    lr = self.learning_rate - self.learning_rate * iter / self.decay_steps
                else:
                    lr = self.learning_rate

                if self.epoch < self.gamma_steps:
                    gamma = self.gamma0 + self.epoch * (self.gamma1 - self.gamma0) / self.gamma_steps
                else:
                    gamma = self.gamma1

                # update Mover
                for it in range(self.num_training_mover):
                    z_batch = self.z_prior.sample([self.batch_size, self.num_z]).astype(np.float32)
                    batch_start, batch_end = batches[batch_idx + it]
                    x_batch = x[batch_start:batch_end]
                    self.tf_session.run(self.m_opt,
                                        feed_dict={self.gamma: gamma, self.lr: lr, self.x: x_batch, self.z: z_batch})

                # update Critic
                for it in range(self.num_training_critic):
                    z_batch = self.z_prior.sample([self.batch_size, self.num_z]).astype(np.float32)
                    batch_start, batch_end = batches[batch_idx + it]
                    x_batch = x[batch_start:batch_end]
                    self.tf_session.run(self.c_opt,
                                        feed_dict={self.gamma: gamma, self.lr: lr, self.x: x_batch, self.z: z_batch})

                # update Generator
                for _ in range(self.num_training_generator):
                    z_batch = self.z_prior.sample([self.batch_size, self.num_z]).astype(np.float32)
                    self.tf_session.run(self.g_opt, feed_dict={self.gamma: gamma, self.lr: lr, self.z: z_batch})

            self._samples(self.samples_fp.format(epoch=self.epoch + 1), num_samples=100)
            idx = np.random.randint(x.shape[0], size=(100 // self.batch_size + 1) * self.batch_size)
            self._samples_h(self.samples_h_fp.format(epoch=self.epoch + 1), x[idx], num_samples=100)
            print('Epoch %d: done.' % (self.epoch + 1))
            self.epoch += 1

    def _create_generator(self, z, train=True, reuse=False, name="generator"):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            normalizer = partial(batch_norm, is_training=train)

            # project to the first layer
            h = linear(z, 4 * 4 * self.num_gen_feature_maps, scope="g_h0.linear")
            h = tf.reshape(h, [self.batch_size, 4, 4, self.num_gen_feature_maps])

            if self.img_size[0] == 32:
                num_res_blocks = 3
            elif self.img_size[0] == 64:
                num_res_blocks = 4
            for i in range(num_res_blocks):
                h = residual_block(h, k=3, s=2, stddev=0.02,
                                   resample="up",
                                   output_dim=self.num_gen_feature_maps,
                                   bn=normalizer,
                                   activation_fn=tf.nn.relu,
                                   name="g_block{}".format(i))

            h = normalizer(h, scope="g_preout.bn")
            h = tf.nn.relu(h, name="g_preout.relu")
            h = tf.layers.conv2d(h, filters=3, kernel_size=3, strides=1, name="g_out.lin", padding="SAME")
            g_out = tf.nn.tanh(h, name="g_out.tanh")
            return g_out

    def _create_critic(self, x, reuse=False, train=True, name="critic"):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            normalizer = partial(batch_norm, is_training=train)

            # residual blocks
            resamples = ["down", "down", None, None]
            h = x
            for i in range(4):
                h = residual_block(h, k=3, s=2, stddev=0.02,
                                   atv_input=i > 0,
                                   bn_input=i > 0,
                                   resample=resamples[i],
                                   output_dim=self.num_cri_feature_maps,
                                   bn=normalizer,
                                   activation_fn=tf.nn.relu,
                                   name="c_block{}".format(i))

            # mean pool layer
            h = normalizer(h, scope="c_mean_pool.bn")
            h = tf.nn.relu(h, name="c_mean_pool.relu")
            h = tf.reduce_mean(h, axis=[1, 2], name="c_mean_pool")

            # output layer
            c_out = linear(h, 1, scope='c_out.lin')
            c_out = self.critic_atv(c_out, name="c_out.atv")
            return c_out

    def _create_mover(self, x, reuse=False, train=True, name="mover"):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            normalizer = partial(batch_norm, is_training=train)

            # residual blocks
            resamples = ["down"] * (self.num_mov_layers // 2) + ["up"] * (self.num_mov_layers // 2)
            h = x
            for i in range(len(resamples)):
                h = residual_block(h, k=3, s=2, stddev=0.02,
                                   atv_input=i > 0,
                                   bn_input=i > 0,
                                   resample=resamples[i],
                                   output_dim=self.num_gen_feature_maps,
                                   bn=normalizer,
                                   activation_fn=tf.nn.relu,
                                   name="m_block{}".format(i))

            h = normalizer(h, scope="m_preout.bn")
            h = tf.nn.relu(h, name="m_preout.relu")
            h = tf.layers.conv2d(h, filters=3, kernel_size=3, strides=1, name="m_out.lin", padding="SAME")
            m_out = tf.nn.tanh(h, name="m_out.tanh")
            return m_out

    def _create_optimizer(self, loss, scope, lr, bt1, bt2):
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        opt = tf.train.AdamOptimizer(lr, beta1=bt1, beta2=bt2)
        grads = opt.compute_gradients(loss, var_list=params)
        train_op = opt.apply_gradients(grads)
        return train_op

    def _generate(self, num_samples=100):
        num = ((num_samples - 1) // self.batch_size + 1) * self.batch_size
        z = self.z_prior.sample([num, self.num_z]).astype(np.float32)
        x = np.zeros([num, self.img_size[0], self.img_size[1], self.img_size[2]], dtype=np.float32)
        batches = make_batches(num, self.batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            z_batch = z[batch_start:batch_end]
            x[batch_start:batch_end] = self.tf_session.run(self.g, feed_dict={self.z: z_batch})
        idx = np.random.permutation(num)[:num_samples]
        return (x[idx] + 1.0) / 2.0


    def _samples(self, filepath, num_samples=100, tile_shape=(10, 10)):
        '''
        Generate samples by Generator
        '''
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        x = self._generate(num_samples)
        imgs = create_image_grid(x, img_size=self.img_size, tile_shape=tile_shape)
        spm.imsave(filepath, imgs)

    def _samples_h(self, filepath, x_batches, num_samples=100, tile_shape=(10, 10)):
        '''
        Strategically move data by Mover
        '''
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        num = ((num_samples - 1) // self.batch_size + 1) * self.batch_size
        x = np.zeros([num, self.img_size[0], self.img_size[1], self.img_size[2]], dtype=np.float32)
        batches = make_batches(num, self.batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            x_batch = x_batches[batch_start:batch_end]
            x[batch_start:batch_end] = self.tf_session.run(self.h, feed_dict={self.x: x_batch})
        idx = np.random.permutation(num)[:num_samples]
        x = (x[idx] + 1.0) / 2.0
        imgs = create_image_grid(x, img_size=self.img_size, tile_shape=tile_shape)
        spm.imsave(filepath, imgs)
