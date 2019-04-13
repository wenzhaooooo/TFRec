import numpy as np
import random
import tensorflow as tf
from data.dataset import Dataset
from model.ModelFactory import ModelFactory
from evaluator import FoldOutEvaluator
np.random.seed(2018)
random.seed(2018)
tf.random.set_random_seed(2018)


if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    model_factory = ModelFactory()
    Model, config = model_factory.get_model()

    dataset = Dataset(config)
    evaluator = FoldOutEvaluator(dataset.train_matrix, dataset.test_matrix)
    model = Model(sess, config, dataset, evaluator)
    model.train_model()
