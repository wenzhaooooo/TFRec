import tensorflow as tf


def log_loss(yij, name="log_loss"):
    with tf.name_scope(name):
        return -tf.log_sigmoid(yij)


def hinge_loss(yij, margin=1.0, name="hinge_loss"):
    with tf.name_scope(name):
        return tf.nn.relu(margin - yij)
