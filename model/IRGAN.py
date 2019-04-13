"""
Paper: IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models
Author: Jun Wang, Lantao Yu, Weinan Zhang, Yu Gong, Yinghui Xu, Benyou Wang, Peng Zhang, and Dell Zhang
"""

import numpy as np
import tensorflow as tf
import pickle
from utils import csr_to_user_dict
from data import DataIterator
from utils import timer
from model.base import AbstractRecommender
from concurrent.futures import ThreadPoolExecutor


class GEN(object):
    def __init__(self, itemNum, userNum, emb_dim, lamda, param=None, initdelta=0.05, learning_rate=0.05):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.g_params = []

        with tf.variable_scope('generator'):
            if self.param == None:
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.userNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.itemNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_bias = tf.Variable(tf.zeros([self.itemNum]))
            else:
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])
                self.item_bias = tf.Variable(param[2])

            self.g_params = [self.user_embeddings, self.item_embeddings, self.item_bias]

        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.reward = tf.placeholder(tf.float32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)
        self.i_bias = tf.gather(self.item_bias, self.i)

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        self.i_prob = tf.gather(
            tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]),
            self.i)

        self.gan_loss = -tf.reduce_mean(tf.log(self.i_prob) * self.reward) + self.lamda * (
            tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.i_bias))

        g_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.gan_updates = g_opt.minimize(self.gan_loss, var_list=self.g_params)

        # for test stage, self.u: [batch_size]
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True) + self.item_bias


class DIS(object):
    def __init__(self, itemNum, userNum, emb_dim, lamda, param=None, initdelta=0.05, learning_rate=0.05):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.d_params = []

        with tf.variable_scope('discriminator'):
            if self.param == None:
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.userNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.itemNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_bias = tf.Variable(tf.zeros([self.itemNum]))
            else:
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])
                self.item_bias = tf.Variable(self.param[2])

        self.d_params = [self.user_embeddings, self.item_embeddings, self.item_bias]

        # placeholder definition
        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.label = tf.placeholder(tf.float32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)
        self.i_bias = tf.gather(self.item_bias, self.i)

        self.pre_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) + self.i_bias
        self.pre_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,
                                                                logits=self.pre_logits) + self.lamda * (
            tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.i_bias)
        )

        d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.d_updates = d_opt.minimize(self.pre_loss, var_list=self.d_params)

        self.reward_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding),
                                           1) + self.i_bias
        self.reward = 2 * (tf.sigmoid(self.reward_logits) - 0.5)

        # for test stage, self.u: [batch_size]
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True) + self.item_bias

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        self.NLL = -tf.reduce_mean(tf.log(
            tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]), self.i))
        )
        # for dns sample
        self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias


class IRGAN(AbstractRecommender):
    def __init__(self, sess, config, dataset, evaluator):
        super(IRGAN, self).__init__(sess, config, dataset, evaluator)
        train_matrix = dataset.train_matrix
        self.users_num, self.items_num = train_matrix.shape

        self.factors_num = config["factors_num"]
        self.lr = config["lr"]
        self.g_reg = config["g_reg"]
        self.d_reg = config["d_reg"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.d_tau = config["d_tau"]
        self.pretrain_file = config["pretrain_file"]

        self.user_pos_train = csr_to_user_dict(train_matrix)

        self.all_items = np.arange(self.items_num)
        self.evaluator = evaluator
        self._build_model()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def _build_model(self):
        with open(self.pretrain_file, "rb") as fin:
            pretrain_params = pickle.load(fin, encoding="latin")
        self.generator = GEN(self.items_num, self.users_num, self.factors_num, self.g_reg, param=pretrain_params,
                             learning_rate=self.lr)
        self.discriminator = DIS(self.items_num, self.users_num, self.factors_num, self.d_reg, param=None,
                                 learning_rate=self.lr)

    # @timer
    def get_train_data(self):
        users_list, items_list, labels_list = [], [], []
        train_users = list(self.user_pos_train.keys())
        with ThreadPoolExecutor() as executor:
            data = executor.map(self.get_train_data_one_user, train_users)
        data = list(data)
        for users, items, labels in data:
            users_list.extend(users)
            items_list.extend(items)
            labels_list.extend(labels)

        dataloader = DataIterator(users_list, items_list, labels_list, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def get_train_data_one_user(self, user):
        user_list, items_list, label_list = [], [], []
        pos = self.user_pos_train[user]

        rating = self.sess.run(self.generator.all_rating, {self.generator.u: [user]})
        rating = np.reshape(rating, [-1])
        rating = np.array(rating) / self.d_tau  # Temperature
        exp_rating = np.exp(rating)
        prob = exp_rating / np.sum(exp_rating)
        neg = np.random.choice(self.all_items, size=len(pos), p=prob)

        user_list.extend([user]*(2*len(pos)))
        items_list.extend(pos)
        label_list.extend([1.0]*len(pos))
        items_list.extend(neg)
        label_list.extend([0.0]*len(neg))
        return user_list, items_list, label_list

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        result = self.evaluate_model()
        self.logger.info("pretrain:\t%s" % result)
        for epoch in range(self.epochs):
            for d_epoch in range(100):
                if d_epoch % 5 == 0:
                    print("d epoch %d" % d_epoch)
                    data_iterator = self.get_train_data()
                self.training_discriminator(data_iterator)
            for g_epoch in range(50):
                self.training_generator()
                result = self.evaluate_model()
                self.logger.info("%d_%d:\t%s" % (epoch, g_epoch, result))

    # @timer
    def training_discriminator(self, data_iterator):
        for users, items, labels in data_iterator:
            feed = {self.discriminator.u: users,
                    self.discriminator.i: items,
                    self.discriminator.label: labels}
            self.sess.run(self.discriminator.d_updates, feed_dict=feed)

    # @timer
    def training_generator(self):
        for user, pos in self.user_pos_train.items():
            sample_lambda = 0.2
            rating = self.sess.run(self.generator.all_logits, {self.generator.u: user})
            exp_rating = np.exp(rating)
            prob = exp_rating / np.sum(exp_rating)  # prob is generator distribution p_\theta

            pn = (1 - sample_lambda) * prob
            pn[pos] += sample_lambda * 1.0 / len(pos)
            # Now, pn is the Pn in importance sampling, prob is generator distribution p_\theta

            sample = np.random.choice(self.all_items, 2 * len(pos), p=pn)
            ###########################################################################
            # Get reward and adapt it with importance sampling
            ###########################################################################
            feed = {self.discriminator.u: user, self.discriminator.i: sample}
            reward = self.sess.run(self.discriminator.reward, feed_dict=feed)
            reward = reward * prob[sample] / pn[sample]
            ###########################################################################
            # Update G
            ###########################################################################
            feed = {self.generator.u: user, self.generator.i: sample, self.generator.reward: reward}
            self.sess.run(self.generator.gan_updates, feed_dict=feed)

    def evaluate_model(self):
        result = self.evaluator.evaluate(self)
        buf = '\t'.join([str(x) for x in result])
        return buf

    def _predict(self):
        # return all ratings matrix
        user_embedding, item_embedding, item_bias = self.sess.run(self.generator.g_params)
        ratings = np.matmul(user_embedding, item_embedding.T) + item_bias
        return ratings
