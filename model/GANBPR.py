"""
using adversarial sampling to train BPR
"""

import tensorflow as tf
import numpy as np
from model.losses import log_loss
from utils.tools import csr_to_user_dict, random_choice, timer
from model.AbstractRecommender import AbstractRecommender
from model.MatrixFactorization import MatrixFactorization
from data.DataIterator import get_data_iterator


class Generator(MatrixFactorization):
    def __init__(self, users_num, items_num, factors_num, params=None, name="generator"):
        super(Generator, self).__init__(users_num, items_num, factors_num, params=params, name=name)

    def forward(self, user, items):
        all_logit = self.get_all_logits(user)
        all_logit = tf.reshape(all_logit, [-1])
        prob = tf.nn.softmax(all_logit)
        items_prob = tf.gather(prob, items)
        return items_prob


class Discriminator(MatrixFactorization):
    def __init__(self, users_num, items_num, factors_num, name="discriminator"):
        super(Discriminator, self).__init__(users_num, items_num, factors_num, name=name)

    def forward(self, users, pos_items, neg_items):
        yi_hat = self.predict(users, pos_items)
        yj_hat = self.predict(users, neg_items)
        loss = log_loss(yi_hat-yj_hat)
        return tf.reduce_sum(loss)

    def get_reward(self, user, pos_items, neg_items):
        loss = self.forward(user, pos_items, neg_items)
        reward = 2 * (tf.sigmoid(loss) - 0.5)
        return reward


class GANBPR(AbstractRecommender):
    def __init__(self, sess, dataset, evaluator):
        super(GANBPR, self).__init__()
        self.logger = self.get_logger(dataset.name)
        train_matrix = dataset.train_matrix
        self.users_num, self.items_num = train_matrix.shape

        self.lr = eval(self.conf["lr"])
        self.batch_size = eval(self.conf["batch_size"])
        self.factors_num = eval(self.conf["factors_num"])
        self.g_reg = eval(self.conf["g_reg"])
        self.d_reg = eval(self.conf["d_reg"])
        self.epochs = eval(self.conf["epochs"])

        self.user_pos_train = csr_to_user_dict(train_matrix)
        self.sess = sess
        self.all_items = np.arange(self.items_num)
        self.evaluator = evaluator
        self.build_model()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        self.user_holder = tf.placeholder(tf.int32, [None, ])
        self.pos_item_holder = tf.placeholder(tf.int32, [None, ])
        self.neg_item_holder = tf.placeholder(tf.int32, [None, ])
        # self.single_user_holder = tf.placeholder(tf.int32, [1, ])
        self.G = Generator(self.users_num, self.items_num, self.factors_num)
        self.D = Discriminator(self.users_num, self.items_num, self.factors_num)
        self.D_predict = self.D.predict(self.user_holder, self.pos_item_holder)

        self.g_user_prob = tf.nn.softmax(self.G.get_all_logits(self.user_holder))

        D_loss = self.D.forward(self.user_holder, self.pos_item_holder, self.neg_item_holder)
        D_reg_loss = self.D.user_l2loss(self.user_holder) + \
                     self.D.item_l2loss(self.pos_item_holder) + \
                     self.D.item_l2loss(self.neg_item_holder)
        D_loss = D_loss + self.d_reg*D_reg_loss
        D_opt = tf.train.GradientDescentOptimizer(self.lr)
        self.D_updates = D_opt.minimize(D_loss, var_list=self.D.parameters())

        neg_item_prob = self.G.forward(self.user_holder, self.neg_item_holder)
        reward = self.D.get_reward(self.user_holder, self.pos_item_holder, self.neg_item_holder)
        G_loss = -tf.reduce_mean(tf.log(neg_item_prob) * reward)
        G_reg_loss = self.G.user_l2loss(self.user_holder) + self.G.item_l2loss(self.neg_item_holder)
        G_loss = G_loss + self.g_reg*G_reg_loss
        G_opt = tf.train.GradientDescentOptimizer(self.lr)
        self.G_updates = G_opt.minimize(G_loss, var_list=self.G.parameters())

    @timer
    def train_generator(self):
        for user, pos_items in self.user_pos_train.items():
            prob = self.sess.run(self.g_user_prob, feed_dict={self.user_holder: [user]})
            prob = np.reshape(prob, newshape=[-1])
            neg_items = random_choice(self.all_items, size=len(pos_items), p=prob)
            feed = {self.user_holder: [user],
                    self.pos_item_holder: pos_items,
                    self.neg_item_holder: neg_items}
            self.sess.run(self.G_updates, feed_dict=feed)

    @timer
    def train_discriminator(self):
        data_iterator = self.generate_data()
        for users, pos_items, neg_items in data_iterator:
            feed = {self.user_holder: users,
                    self.pos_item_holder: pos_items,
                    self.neg_item_holder: neg_items}
            self.sess.run(self.D_updates, feed_dict=feed)

    def generate_data(self):
        users_list, pos_items_list, neg_items_list = [], [], []
        for user, pos_items in self.user_pos_train.items():
            prob = self.sess.run(self.g_user_prob, feed_dict={self.user_holder: [user]})
            prob = np.reshape(prob, newshape=[-1])
            neg_items = random_choice(self.all_items, size=len(pos_items), p=prob, exclusion=pos_items)

            users_list.extend([user]*len(pos_items))
            pos_items_list.extend(pos_items)
            neg_items_list.extend(neg_items)

        return get_data_iterator(users_list, pos_items_list, neg_items_list,
                                 batch_size=self.batch_size, shuffle=True)

    def train_model(self):
        self.evaluate_model("init")
        for epoch in range(self.epochs):
            self.train_discriminator()
            self.train_generator()
            self.evaluate_model(epoch)

    def evaluate_model(self, epoch):
        epoch = str(epoch)
        result = self.evaluator.evaluate(self)
        buf = '\t'.join([str(x) for x in result])
        self.logger.info("epoch %s, G:\t%s" % (epoch, buf))

    def predict_for_eval(self, user=None, items=None):
        if user is None:  # return all ratings matrix
            user_embedding, item_embedding, item_bias = self.sess.run(self.D.parameters())
            ratings = np.matmul(user_embedding, item_embedding.T) + item_bias
        elif items is None:  # return all ratings of one user
            raise NotImplementedError  # TODO
        else:  # return the given items rating about the given user.
            raise NotImplementedError  # TODO
        return ratings

