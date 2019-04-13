import numpy as np
import tensorflow as tf
from model.base import AbstractRecommender
from model.base import MatrixFactorization
from utils import random_choice
from model.losses import log_loss
from data import DataIterator
from utils import csr_to_user_dict
from concurrent.futures import ThreadPoolExecutor
import pickle


class DNSBPR(AbstractRecommender):
    def __init__(self, sess, config, dataset, evaluator):
        super(DNSBPR, self).__init__(sess, config, dataset, evaluator)
        self.config = config
        train_matrix = dataset.train_matrix
        self.users_num, self.items_num = train_matrix.shape

        self.factors_num = config["factors_num"]
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.neg_num = config["neg_num"]
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

        self.all_logits_tensor = self.mf.all_logits(self.user_h)

    def get_training_data(self):
        users_list = []
        pos_items_list = []
        neg_items_list = []
        users = self.user_pos_train.keys()
        with ThreadPoolExecutor() as executor:
            batch_result = executor.map(self._get_neg_items, users)
        for user, pos, neg in batch_result:
            users_list.extend(user)
            pos_items_list.extend(pos)
            neg_items_list.extend(neg)

        return DataIterator(users_list, pos_items_list, neg_items_list, batch_size=self.batch_size, shuffle=True)

    def _get_neg_items(self, user):
        pos_item = self.user_pos_train[user]
        pos_len = len(pos_item)
        feed = {self.user_h: [user]}
        logits = self.sess.run(self.all_logits_tensor, feed_dict=feed)
        logits = np.reshape(logits, newshape=[-1])

        neg_pool = random_choice(self.all_items, size=self.neg_num * pos_len, exclusion=pos_item)

        neg_logits = logits[neg_pool]

        neg_pool = np.reshape(neg_pool, newshape=[pos_len, self.neg_num])
        neg_logits = np.reshape(neg_logits, newshape=[pos_len, self.neg_num])

        neg_item = neg_pool[np.arange(pos_len), np.argmax(neg_logits, axis=1)]
        return [user]*pos_len, pos_item, neg_item

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
        self.save_model()

    def _predict(self):
        # return all ratings matrix
        user_embedding, item_embedding, item_bias = self.sess.run(self.mf.parameters())
        all_ratings = np.matmul(user_embedding, item_embedding.T) + item_bias
        return all_ratings

    def evaluate_model(self):
        result = self.evaluator.evaluate(self)
        buf = '\t'.join([str(x) for x in result])
        return buf

    def save_model(self):
        params = self.sess.run(self.mf.parameters())
        with open("%s_dnsbpr.pkl" % self.config["data_name"], "wb") as fout:
            pickle.dump(params, fout)

