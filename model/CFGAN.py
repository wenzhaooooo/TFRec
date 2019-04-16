"""
Paper: CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks
Author: Dong-Kyu Chae, Jin-Soo Kang, Sang-Wook Kim and Jung-Tae Lee
"""
# TODO Turning

from model.base import AbstractRecommender
import numpy as np
from utils.tools import csr_to_user_dict, random_choice
import tensorflow as tf
from data import DataIterator


class CFGAN(AbstractRecommender):
    def __init__(self, sess, config, dataset, evaluator):
        super(CFGAN, self).__init__(sess, config, dataset, evaluator)
        train_matrix = dataset.train_matrix
        self.users_num, self.items_num = train_matrix.shape

        self.g_lr = config["g_lr"]
        self.d_lr = config["d_lr"]
        self.d_hidden_units = config["d_hidden_units"]
        self.g_hidden_units = config["g_hidden_units"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]

        self.s_zr = config["s_zr"]
        self.s_pm = config["s_pm"]
        self.alpha = config["alpha"]

        self.user_pos_train = csr_to_user_dict(train_matrix)
        self.all_items = np.arange(self.items_num)
        self.evaluator = evaluator
        self.train_matrix_dense = train_matrix.todense()

        self.g_layers = None
        self.d_layers = None

        self._build_model()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def generator(self, input):
        if self.g_layers is None:
            self.g_layers = []
            xavier_init = tf.contrib.layers.xavier_initializer()
            for units in self.g_hidden_units:
                hidden_layer = tf.layers.Dense(units, activation=tf.sigmoid, kernel_initializer=xavier_init)
                self.g_layers.append(hidden_layer)

            output_layer = tf.layers.Dense(self.items_num, activation=tf.sigmoid, kernel_initializer=xavier_init)
            self.g_layers.append(output_layer)

        for layer in self.g_layers:
            input = layer.apply(input)
        return input

    def discriminator(self, input):
        if self.d_layers is None:
            self.d_layers = []
            xavier_init = tf.contrib.layers.xavier_initializer()
            for units in self.d_hidden_units:
                hidden_layer = tf.layers.Dense(units, activation=tf.sigmoid, kernel_initializer=xavier_init)
                self.d_layers.append(hidden_layer)

            output_layer = tf.layers.Dense(1, activation=tf.sigmoid, kernel_initializer=xavier_init)
            self.d_layers.append(output_layer)

        for layer in self.d_layers:
            input = layer.apply(input)
        return input

    def _build_model(self):
        self.user_context = tf.placeholder(tf.float32, shape=[None, self.items_num])

        with tf.variable_scope("Generator"):
            self.r_hat = self.generator(self.user_context)

        with tf.variable_scope("masking"):
            self.mask_h = tf.placeholder(tf.float32, shape=[None, self.items_num], name='mask')
            self.N_zr_h = tf.placeholder(tf.float32, shape=[None, self.items_num], name='N_zr')

            fake_sample = tf.multiply(self.r_hat, self.mask_h)

        with tf.variable_scope("Discriminator"):
            d_real = tf.concat([self.user_context, self.user_context], 1)
            real_prob = self.discriminator(d_real)

            d_fake = tf.concat([self.user_context, fake_sample], 1)
            fake_prob = self.discriminator(d_fake)

        with tf.variable_scope("loss"):
            d_loss = tf.log(real_prob+1e-7) + tf.log(1-fake_prob+1e-7)
            d_loss = -tf.reduce_mean(d_loss)

            g_loss = tf.log(1-fake_prob+1e-7) + self.alpha * tf.nn.l2_loss(tf.multiply(self.N_zr_h, fake_sample))
            g_loss = tf.reduce_mean(g_loss)

        with tf.variable_scope("update"):
            d_params = []
            for layer in self.d_layers:
                d_params.extend(layer.variables)
            g_params = []
            for layer in self.g_layers:
                g_params.extend(layer.variables)
            self.d_opt = tf.train.AdamOptimizer(learning_rate=self.d_lr).minimize(d_loss, var_list=d_params)
            self.g_opt = tf.train.AdamOptimizer(learning_rate=self.g_lr).minimize(g_loss, var_list=g_params)

    def train_model(self):
        for epoch in range(self.epochs):
            data = self.get_train_data()
            for user, mask, N_zr in data:
                self.sess.run(self.d_opt, feed_dict={self.user_context: user, self.mask_h: mask, self.N_zr_h: N_zr})

                self.sess.run(self.g_opt, feed_dict={self.user_context: user, self.mask_h: mask, self.N_zr_h: N_zr})

            result = self.evaluate_model()
            self.logger.info("%d:\t%s" % (epoch, result))

    def get_train_data(self):
        self._mask = np.zeros([self.users_num, self.items_num])
        self._N_zr = np.zeros([self.users_num, self.items_num])

        for user, pos_items in self.user_pos_train.items():
            pos_items = self.user_pos_train[user]
            self._mask[user][pos_items] = 1
            neg = random_choice(self.all_items, size=int(self.s_pm * self.items_num), replace=False,
                                exclusion=pos_items)
            self._mask[user][neg] = 1

            neg = random_choice(self.all_items, size=int(self.s_zr * self.items_num), replace=False,
                                exclusion=pos_items)
            self._N_zr[user][neg] = 1

        return DataIterator(self.train_matrix_dense.tolist(), self._mask, self._N_zr, batch_size=self.batch_size, shuffle=True)

    def evaluate_model(self):
        result = self.evaluator.evaluate(self)
        buf = '\t'.join([str(x) for x in result])
        return buf

    def _predict(self):
        all_context = DataIterator(self.train_matrix_dense.tolist(), batch_size=self.batch_size)
        all_rating = []
        for users in all_context:
            r_hat = self.sess.run(self.r_hat, feed_dict={self.user_context: users})
            all_rating.extend(r_hat)

        return np.array(all_rating)
