from model.AbstractRecommender import AbstractRecommender
from model.MatrixFactorization import MatrixFactorization
import numpy as np
import tensorflow as tf
from utils.tools import random_choice
from model.losses import log_loss
from data.DataLoader import get_data_loader
import configparser
from evaluation.Evaluator import FoldOutEvaluator


class AOBPR(AbstractRecommender):
    def __init__(self, sess, dataset):
        super(AOBPR, self).__init__()
        config = configparser.ConfigParser()
        config.read("conf/AOBPR.ini")
        self.conf = dict(config.items("hyperparameters"))
        print("AOBPR arguments: %s " % (self.conf))
        train_matrix = dataset.train_matrix
        valid_matrix = dataset.valid_matrix
        test_matrix = dataset.test_matrix

        self.users_num, self.items_num = train_matrix.shape

        self.factors_num = int(self.conf["factors_num"])
        self.lr = float(self.conf["lr"])
        self.reg = float(self.conf["reg"])
        self.epochs = int(self.conf["epochs"])
        self.batch_size = int(self.conf["batch_size"])
        self.user_pos_train = {}
        self.user_pos_test = {}
        for u in range(self.users_num):
            self.user_pos_train[u] = train_matrix.getrow(u).indices
            self.user_pos_test[u] = test_matrix.getrow(u).indices
        self.all_items = np.arange(self.items_num)
        self.evaluator = FoldOutEvaluator(train_matrix, valid_matrix, test_matrix)

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
        self.total_loss = loss + self.reg*reg_loss
        opt = tf.train.GradientDescentOptimizer(self.lr)
        self.update = opt.minimize(self.total_loss, name="update")
        self.all_logits = self.mf.get_all_logits(self.user_h)

    def get_training_data(self):
        users = []
        pos_items = []
        neg_items = []
        for u, pos in self.user_pos_train.items():
            pos_len = len(pos)
            feed = {self.user_h: [u]}
            logits = self.sess.run(self.all_logits, feed_dict=feed)

            neg_exp = np.exp(logits)
            neg_prob = neg_exp / np.sum(neg_exp)
            neg = random_choice(self.all_items, size=pos_len, exclusion=pos, p=neg_prob)

            users += [u]*pos_len
            pos_items += pos.tolist()
            neg_items += neg.tolist()

        return get_data_loader(users, pos_items, neg_items, batch_size=self.batch_size, shuffle=True)

    def train_model(self):
        self.evaluator.print_metrics()
        for epoch in range(self.epochs):
            train_data = self.get_training_data()
            for users, pos_items, neg_items in train_data:
                feed = {self.user_h: users,
                        self.pos_item_h: pos_items,
                        self.neg_item_h: neg_items}
                self.sess.run(self.update, feed_dict=feed)
            result = self.eval()
            buf = '\t'.join([str(x) for x in result])
            print("epoch %d:\t\t%s" % (epoch, buf))

    def get_ratings_matrix(self):
        user_embedding, item_embedding, item_bias = self.sess.run(self.mf.parameters())
        all_ratings = np.matmul(user_embedding, item_embedding.T) + item_bias
        return all_ratings

    def eval(self):
        valid_result, test_result = self.evaluator.evaluate(self)
        return test_result
