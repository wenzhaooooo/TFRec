from model.AbstractRecommender import AbstractRecommender
from model.MatrixFactorization import MatrixFactorization
import tensorflow as tf
import pickle
import numpy as np
from utils.tools import csr_to_user_dict
from data.DataIterator import get_data_iterator
from utils.tools import timer


class GEN():
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


class DIS():
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
    # TODO
    def __init__(self, sess, dataset, evaluator):
        super(IRGAN, self).__init__()
        self.logger = self.get_logger(dataset.name)
        train_matrix = dataset.train_matrix
        self.users_num, self.items_num = train_matrix.shape

        self.factors_num = eval(self.conf["factors_num"])
        self.lr = eval(self.conf["lr"])
        self.g_reg = eval(self.conf["g_reg"])
        self.d_reg = eval(self.conf["d_reg"])
        self.epochs = eval(self.conf["epochs"])
        self.g_epoch = eval(self.conf["g_epoch"])
        self.d_epoch = eval(self.conf["d_epoch"])
        self.batch_size = eval(self.conf["batch_size"])
        self.pretrain_file = self.conf["pretrain_file"]

        self.user_pos_train = csr_to_user_dict(train_matrix)

        self.all_items = np.arange(self.items_num)
        self.evaluator = evaluator
        self.build_model()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        with open(self.pretrain_file, "rb") as fin:
            pretrain_params = pickle.load(fin, encoding="latin")
        self.generator = GEN(self.items_num, self.users_num, self.factors_num, self.g_reg, param=pretrain_params,
                             learning_rate=self.lr)
        self.discriminator = DIS(self.items_num, self.users_num, self.factors_num, self.d_reg, param=None,
                                 learning_rate=self.lr)

    @timer
    def get_train_data(self):
        users, items, labels = [], [], []
        for u in self.user_pos_train:
            pos = self.user_pos_train[u]

            rating = self.sess.run(self.generator.all_rating, {self.generator.u: [u]})
            rating = np.reshape(rating, [-1])
            rating = np.array(rating) / 0.2  # Temperature
            exp_rating = np.exp(rating)
            prob = exp_rating / np.sum(exp_rating)

            neg = np.random.choice(self.all_items, size=len(pos), p=prob)
            for i, j in zip(pos, neg):
                users.append(u)
                items.append(i)
                labels.append(1.0)

                users.append(u)
                items.append(j)
                labels.append(0.0)
        dataloader = get_data_iterator(users, items, labels, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        self.evaluate_model("pretrain")
        for epoch in range(self.epochs):
            data_iterator = self.get_train_data()
            for d_epoch in range(self.d_epoch):
                if d_epoch % 5 == 0:
                    print("d epoch %d" % d_epoch)
                    data_iterator = self.get_train_data()
                self.training_discriminator(data_iterator)
            for g_epoch in range(self.g_epoch):
                self.training_generator()
                self.evaluate_model("%d_%d" % (epoch, g_epoch))

    @timer
    def training_discriminator(self, data_iterator):
        for users, items, labels in data_iterator:
            feed = {self.discriminator.u: users,
                    self.discriminator.i: items,
                    self.discriminator.label: labels}
            self.sess.run(self.discriminator.d_updates, feed_dict=feed)

    @timer
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

    def evaluate_model(self, epoch):
        epoch = str(epoch)
        result = self.evaluator.evaluate(self)
        buf = '\t'.join([str(x) for x in result])
        self.logger.info("epoch %s, G:\t%s" % (epoch, buf))

    def predict_for_eval(self, user=None, items=None):
        if user is None:  # return all ratings matrix
            user_embedding, item_embedding, item_bias = self.sess.run(self.generator.g_params)
            ratings = np.matmul(user_embedding, item_embedding.T) + item_bias
        elif items is None:  # return all ratings of one user
            raise NotImplementedError  # TODO
        else:  # return the given items rating about the given user.
            raise NotImplementedError  # TODO
        return ratings