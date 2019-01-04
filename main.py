import numpy as np
from model.BPR import BPR
import random
import tensorflow as tf
from data.DataSplitter import GivenRatioDataSplitter
# from tensorflow.python import debug as tf_debug
np.random.seed(2018)
random.seed(2018)
tf.random.set_random_seed(2018)


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # sess = tf.InteractiveSession(config=config)
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "sun:7000")

    splitter = GivenRatioDataSplitter(sep="\t", data_format="UIRT")
    dataset = splitter.load_data(r"dataset/ml100k.all")
    model = BPR(sess, dataset)
    model.train_model()
