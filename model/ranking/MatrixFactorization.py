import tensorflow as tf


class MatrixFactorization(object):
    def __init__(self, users_num, items_num, factors_num, params=None, name="MF"):
        self.users_num = users_num
        self.items_num = items_num
        self.factors_num = factors_num
        # self.params = params

        # initialize parameters
        with tf.name_scope(name):
            if params is not None:
                self.user_embeddings = tf.Variable(params[0], name="user_embedding")
                self.item_embeddings = tf.Variable(params[1], name="item_embedding")
                self.item_biases = tf.Variable(params[2], name="item_bias")
            else:
                self.user_embeddings = tf.Variable(tf.random_uniform([self.users_num, self.factors_num],
                                                                     minval=-0.05, maxval=0.05), name="user_embedding")
                self.item_embeddings = tf.Variable(tf.random_uniform([self.items_num, self.factors_num],
                                                                     minval=-0.05, maxval=0.05), name="item_embedding")
                self.item_biases = tf.Variable(tf.zeros([self.items_num]), name="item_bias")

    def parameters(self):
        return [self.user_embeddings, self.item_embeddings, self.item_biases]

    def predict(self, users, items, name="predict"):
        with tf.name_scope(name):
            user_embedding = tf.nn.embedding_lookup(self.user_embeddings, users)
            item_embedding = tf.nn.embedding_lookup(self.item_embeddings, items)
            item_bias = tf.gather(self.item_biases, items)
            ratings = tf.reduce_sum(tf.multiply(user_embedding, item_embedding), axis=1) + item_bias
            return ratings

    def get_all_logits(self, users=None, name="all_logits"):
        # If user is 'None', then return all users ratings
        with tf.name_scope(name):
            if users is not None:
                user_embedding = tf.nn.embedding_lookup(self.user_embeddings, users)
                all_rating = tf.matmul(user_embedding, self.item_embeddings, transpose_b=True) + self.item_biases
            else:
                all_rating = tf.matmul(self.user_embeddings, self.item_embeddings, transpose_b=True) + self.item_biases
        return all_rating

    def user_l2loss(self, users, name="user_l2loss"):
        with tf.name_scope(name):
            user_embedding = tf.nn.embedding_lookup(self.user_embeddings, users)
            return tf.nn.l2_loss(user_embedding)

    def item_l2loss(self, items, name="item_l2loss"):
        with tf.name_scope(name):
            item_embedding = tf.nn.embedding_lookup(self.item_embeddings, items)
            item_bias = tf.gather(self.item_biases, items)
            return tf.nn.l2_loss(item_embedding) + tf.nn.l2_loss(item_bias)
