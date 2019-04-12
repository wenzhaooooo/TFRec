import tensorflow as tf


def log_loss(yij, epsilon=1e-7, name="log_loss"):
    """ bpr loss
    :param yij:
    :param epsilon: A small increment to add to avoid taking a log of zero.
    :param name:
    :return:
    """
    with tf.name_scope(name):
        return -tf.log_sigmoid(yij+epsilon)


def hinge_loss(yij, margin=1.0, name="hinge_loss"):
    with tf.name_scope(name):
        return tf.nn.relu(margin - yij)
