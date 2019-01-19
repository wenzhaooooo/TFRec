import numpy as np
import random
import tensorflow as tf
import configparser
from data.DataLoader import DataLoader
from model.ModelFactory import ModelFactory
from evaluation.Evaluator import LeaveOneOutEvaluator, FoldOutEvaluator
# from tensorflow.python import debug as tf_debug
np.random.seed(2018)
random.seed(2018)
tf.random.set_random_seed(2018)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("abcrec.ini")
    data_info = dict(config.items("data"))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #sess = tf.InteractiveSession(config=config)
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "sun:7000")
    model_factory = ModelFactory()
    print("Available models: %s" % model_factory.available_model())
    Model = model_factory.get_model("ganbpr")

    data_loader = DataLoader(data_info)
    dataset = data_loader.load_data()
    valid_evaluator = FoldOutEvaluator(dataset.train_matrix, dataset.valid_matrix, dataset.test_negative)
    test_evaluator = FoldOutEvaluator(dataset.train_matrix, dataset.test_matrix, dataset.test_negative)
    model = Model(sess, dataset, valid_evaluator)
    model.train_model()
