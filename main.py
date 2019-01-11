import numpy as np
import random
import tensorflow as tf
import configparser
from data.DataLoader import DataLoader
from model.ModelFactory import ModelFactory
# from tensorflow.python import debug as tf_debug
np.random.seed(2018)
random.seed(2018)
tf.random.set_random_seed(2018)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("abcrec.ini")
    conf = dict(config.items("default"))
    data_input_path = conf["data.input.path"]
    dataset_name = conf["data.input.dataset"]
    splitter = conf["data.splitter"]
    separator = eval(conf["data.convert.separator"])
    recommender = str(conf["recommender"])
    evaluate_neg = int(conf["rec.evaluate.neg"])
    is_save = bool(conf["save.weight"])
    save_path = str(conf["save.path"])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # sess = tf.InteractiveSession(config=config)
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "sun:7000")
    model_factory = ModelFactory()
    print("Available models: %s" % model_factory.available_model())
    Model = model_factory.get_model("BPR")

    data_loader = DataLoader(data_input_path+dataset_name, splitter, separator, evaluate_neg, dataset_name)
    dataset = data_loader.load_data()
    model = Model(sess, dataset)
    model.train_model()
