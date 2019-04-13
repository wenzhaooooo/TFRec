"""
Paper: Adversarial Personalized Ranking for Recommendation
Author: Xiangnan He, Zhankui He, Xiaoyu Du and Tat-Seng Chua
"""

import tensorflow as tf
import pickle
import numpy as np
from model.base import AbstractRecommender
from utils.tools import csr_to_user_dict
from utils.tools import random_choice
from data import DataIterator
from utils.tools import timer
from concurrent.futures import ThreadPoolExecutor


class APR(AbstractRecommender):
    def __init__(self, sess, config, dataset, evaluator):
        super(APR, self).__init__(sess, config, dataset, evaluator)
        train_matrix = dataset.train_matrix
        self.users_num, self.items_num = train_matrix.shape

        self.factors_num = config["factors_num"]
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.reg_adv = config["reg_adv"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.eps = config["eps"]
        self.pretrain_file = config["pretrain_file"]

        self.user_pos_train = csr_to_user_dict(train_matrix)

        self.all_items = np.arange(self.items_num)
        self.evaluator = evaluator
        self._build_model()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_neg")

    def _create_variables(self):
        with open(self.pretrain_file, "rb") as fin:
            param = pickle.load(fin, encoding="latin")

        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(param[0], name='embedding_P')
            self.embedding_Q = tf.Variable(param[1], name='embedding_Q')
            self.embedding_B = tf.Variable(param[2], name='embedding_B')

            self.delta_P = tf.Variable(tf.zeros(shape=[self.users_num, self.factors_num]),
                                       name='delta_P', dtype=tf.float32, trainable=False)  # (users, embedding_size)
            self.delta_Q = tf.Variable(tf.zeros(shape=[self.items_num, self.factors_num]),
                                       name='delta_Q', dtype=tf.float32, trainable=False)  # (items, embedding_size)
            self.delta_B = tf.Variable(tf.zeros(shape=[self.items_num]),
                                       name='delta_B', dtype=tf.float32, trainable=False)  # (items)

            self.h = tf.constant(1.0, tf.float32, [self.factors_num, 1], name="h")

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # embedding look up
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input), 1)  # (b, embedding_size)
            self.embedding_b = tf.gather(self.embedding_B, item_input)
            return tf.matmul(self.embedding_p * self.embedding_q, self.h) + self.embedding_b, \
                   self.embedding_p, self.embedding_q, self.embedding_b  # (b, embedding_size) * (embedding_size, 1)

    def _create_inference_adv(self, item_input):
        with tf.name_scope("inference_adv"):
            # embedding look up
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input), 1)  # (b, embedding_size)
            self.embedding_b = tf.gather(self.embedding_B, item_input)
            # add adversarial noise
            self.P_plus_delta = self.embedding_p + tf.reduce_sum(tf.nn.embedding_lookup(self.delta_P, self.user_input), 1)
            self.Q_plus_delta = self.embedding_q + tf.reduce_sum(tf.nn.embedding_lookup(self.delta_Q, item_input), 1)
            self.B_plus_delta = self.embedding_b + tf.gather(self.delta_B, item_input)
            return tf.matmul(self.P_plus_delta * self.Q_plus_delta, self.h) + self.B_plus_delta, \
                   self.embedding_p, self.embedding_q, self.embedding_b  # (b, embedding_size) * (embedding_size, 1)

    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            self.output, embed_p_pos, embed_q_pos, embed_b_pos = self._create_inference(self.item_input_pos)
            self.output_neg, embed_p_neg, embed_q_neg, embed_b_neg = self._create_inference(self.item_input_neg)
            self.result = tf.clip_by_value(self.output - self.output_neg, -80.0, 1e8)
            # self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result))) # this is numerically unstable
            self.loss = tf.reduce_sum(tf.nn.softplus(-self.result))

            # loss to be omptimized
            self.opt_loss = self.loss + self.reg * tf.reduce_mean(tf.square(embed_p_pos) +
                                                                  tf.square(embed_q_pos) +
                                                                  tf.square(embed_q_neg) +
                                                                  tf.square(embed_b_pos) +
                                                                  tf.square(embed_b_neg))  # embed_p_pos == embed_q_neg

            # loss for L(Theta + adv_Delta)
            self.output_adv, embed_p_pos, embed_q_pos, embed_b_pos = self._create_inference_adv(self.item_input_pos)
            self.output_neg_adv, embed_p_neg, embed_q_neg, embed_b_neg = self._create_inference_adv(self.item_input_neg)
            self.result_adv = tf.clip_by_value(self.output_adv - self.output_neg_adv, -80.0, 1e8)
            # self.loss_adv = tf.reduce_sum(tf.log(1 + tf.exp(-self.result_adv)))
            self.loss_adv = tf.reduce_sum(tf.nn.softplus(-self.result_adv))
            self.opt_loss += self.reg_adv * self.loss_adv + \
                             self.reg * tf.reduce_mean(tf.square(embed_p_pos) +
                                                       tf.square(embed_q_pos) +
                                                       tf.square(embed_q_neg) +
                                                       tf.square(embed_b_pos) +
                                                       tf.square(embed_b_neg)
                                                       )

    def _create_adversarial(self):
        with tf.name_scope("adversarial"):

            # generate the adversarial weights by gradient-based method
            # return the IndexedSlice Data: [(values, indices, dense_shape)]
            # grad_var_P: [grad,var], grad_var_Q: [grad, var]
            self.grad_P, self.grad_Q, self.grad_B = tf.gradients(self.loss, [self.embedding_P, self.embedding_Q, self.embedding_B])

            # convert the IndexedSlice Data to Dense Tensor
            self.grad_P_dense = tf.stop_gradient(self.grad_P)
            self.grad_Q_dense = tf.stop_gradient(self.grad_Q)
            self.grad_B_dense = tf.stop_gradient(self.grad_B)

            # normalization: new_grad = (grad / |grad|) * eps
            self.update_P = self.delta_P.assign(tf.nn.l2_normalize(self.grad_P_dense, 1) * self.eps)
            self.update_Q = self.delta_Q.assign(tf.nn.l2_normalize(self.grad_Q_dense, 1) * self.eps)
            self.update_B = self.delta_B.assign(tf.nn.l2_normalize(self.grad_B_dense) * self.eps)

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr).minimize(self.opt_loss)

    def _build_model(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
        self._create_adversarial()

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        result = self.evaluate_model()
        self.logger.info("pretrain:\t%s" % result)

        for epoch in range(self.epochs):
            self.train_model_epoch()
            result = self.evaluate_model()
            self.logger.info("%d:\t%s" % (epoch, result))

    # @timer
    def train_model_epoch(self):
        data = self.get_train_data()
        for users, pos_items, neg_items in data:
            feed_dict = {self.user_input: np.array(users, dtype=np.int32)[:, None],
                         self.item_input_pos: np.array(pos_items)[:, None],
                         self.item_input_neg: np.array(neg_items)[:, None]}
            # for BPR training
            self.sess.run([self.update_P, self.update_Q, self.update_B], feed_dict)
            self.sess.run(self.optimizer, feed_dict)

    def get_train_data(self):
        users_list, pos_items, neg_items = [], [], []
        train_users = list(self.user_pos_train.keys())
        with ThreadPoolExecutor() as executor:
            data = executor.map(self.get_train_data_one_user, train_users)
        data = list(data)
        for users, pos, neg in data:
            users_list.extend(users)
            pos_items.extend(pos)
            neg_items.extend(neg)

        dataloader = DataIterator(users_list, pos_items, neg_items, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def get_train_data_one_user(self, user):
        pos = self.user_pos_train[user]
        pos_len = len(pos)
        neg = random_choice(self.all_items, size=pos_len, exclusion=pos)
        return [user] * pos_len, pos.tolist(), neg.tolist()

    def evaluate_model(self):
        result = self.evaluator.evaluate(self)
        buf = '\t'.join([str(x) for x in result])
        return buf

    def _predict(self):
        # return all ratings matrix
        user_embedding, item_embedding, item_bias = self.sess.run([self.embedding_P, self.embedding_Q, self.embedding_B])
        ratings = np.matmul(user_embedding, item_embedding.T) + item_bias
        return ratings
