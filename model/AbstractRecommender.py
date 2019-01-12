import configparser
from utils.logger import Logger
import os


class AbstractRecommender(object):
    def __init__(self):
        self.logger = None
        config = configparser.ConfigParser()
        config.read("conf/%s.ini" % self.__class__.__name__)
        self.conf = dict(config.items("hyperparameters"))

    def build_model(self):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError
    
    def evaluate_model(self, epoch):
        raise NotImplementedError

    def get_logger(self, data_name):
        log_dir = os.path.join("./Log", data_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logger_name = '_'.join(["{}={}".format(arg, value) for arg, value in sorted(self.conf.items())
                                if len(value) < 10])
        logger_name = "%s_log_%s.log" % (self.__class__.__name__, logger_name)
        logger_name = os.path.join(log_dir, logger_name)
        logger = Logger(logger_name)
        logger.info("\n")
        logger.info("Recommender:%s" % self.__class__.__name__)
        argument = '\n'.join(["{}={}".format(arg, value) for arg, value in sorted(self.conf.items())])
        logger.info("\nHyperparameters:\n%s " % argument)
        return logger
