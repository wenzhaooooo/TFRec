"""
Paper: BPR: Bayesian Personalized Ranking from Implicit Feedback
Author: Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme
"""

import numpy as np
import tensorflow as tf
from model.base import AbstractRecommender
from model.base import MatrixFactorization
from utils import random_choice
from model.losses import log_loss
from data import DataIterator
from utils import csr_to_user_dict


class BPR(AbstractRecommender):
    def __init__(self, sess, config, dataset, evaluator):
        super(BPR, self).__init__(sess, config, dataset, evaluator)
        train_matrix = dataset.train_matrix
        self.users_num, self.items_num = train_matrix.shape

        self.factors_num = config["factors_num"]
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.user_pos_train = csr_to_user_dict(train_matrix)
        self.all_items = np.arange(self.items_num)
        self.evaluator = evaluator

        self.mf = MatrixFactorization(self.users_num, self.items_num, self.factors_num, name=self.__class__.__name__)
        self._build_model()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def _build_model(self):
        self.user_h = tf.placeholder(tf.int32, name="user")
        self.pos_item_h = tf.placeholder(tf.int32, name="pos_item")
        self.neg_item_h = tf.placeholder(tf.int32, name="neg_item")
        yi_hat = self.mf.predict(self.user_h, self.pos_item_h, name="yi_hat")
        yj_hat = self.mf.predict(self.user_h, self.neg_item_h, name="yj_hat")
        loss = tf.reduce_sum(log_loss(yi_hat-yj_hat))
        reg_loss = self.mf.user_l2loss(self.user_h, name="user_reg") + \
                   self.mf.item_l2loss(self.pos_item_h, name="pos_item_reg") + \
                   self.mf.item_l2loss(self.neg_item_h, name="neg_item_reg")
        self.final_loss = loss + self.reg * reg_loss
        opt = tf.train.GradientDescentOptimizer(self.lr)
        self.update = opt.minimize(self.final_loss, name="update")

    def get_training_data(self):
        users = []
        pos_items = []
        neg_items = []
        for u, pos in self.user_pos_train.items():
            pos_len = len(pos)
            neg = random_choice(self.all_items, size=pos_len, exclusion=pos)

            users.extend([u]*pos_len)
            pos_items.extend(pos.tolist())
            neg_items.extend(neg.tolist())

        return DataIterator(users, pos_items, neg_items, batch_size=self.batch_size, shuffle=True)

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            train_data = self.get_training_data()
            for users, pos_items, neg_items in train_data:
                feed = {self.user_h: users,
                        self.pos_item_h: pos_items,
                        self.neg_item_h: neg_items}
                self.sess.run(self.update, feed_dict=feed)
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def _predict(self):
        # return all ratings matrix
        user_embedding, item_embedding, item_bias = self.sess.run(self.mf.parameters())
        all_ratings = np.matmul(user_embedding, item_embedding.T) + item_bias
        return all_ratings

    def evaluate_model(self):
        result = self.evaluator.evaluate(self)
        buf = '\t'.join([str(x) for x in result])
        return buf
