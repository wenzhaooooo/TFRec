import numpy as np
import random
import tensorflow as tf
from data.DataSplitter import GivenRatioDataSplitter, LeaveOneOutDataSplitter
from model.ModelFactory import ModelFactory
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
    model_factory = ModelFactory()
    print("Available models: %s" % model_factory.available_model())
    Model = model_factory.create_model("AOBPR")

    splitter = LeaveOneOutDataSplitter(sep="\t", data_format="UIRT", negative_num=500)
    dataset = splitter.load_data(r"dataset/ml100k.all")
    model = Model(sess, dataset)
    model.train_model()
