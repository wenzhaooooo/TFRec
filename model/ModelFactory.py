from model.BPR import BPR
from model.DNSBPR import DNSBPR
from model.Pop import Pop
from model.IRGAN import IRGAN
from model.APL import APL
from model.APR import APR
from model.CFGAN import CFGAN
import configparser
import os
from utils import Logger
from collections import OrderedDict
import time


class ModelFactory(object):
    def __init__(self):
        self.model_dict = {"BPR": BPR,
                           "DNSBPR": DNSBPR,
                           "Pop": Pop,
                           "IRGAN": IRGAN,
                           "APL": APL,
                           "APR": APR,
                           "CFGAN": CFGAN
                           }

    def get_model(self):
        # read config
        tfrec_config, model_config = self._read_config()
        # select model
        model_name = tfrec_config["model"]
        Model = self.model_dict[model_name]
        # create logger
        Model.logger = self._create_logger(tfrec_config, model_config)
        # get parameters
        config = self._eval_parameter(tfrec_config, model_config)

        return Model, config

    def _read_config(self):
        config = configparser.ConfigParser()
        config.read("tfrec.ini")
        tfrec_config = OrderedDict(config._sections["tfrec"].items())
        model_name = tfrec_config["model"]

        model_config_path = os.path.join("./conf", model_name + ".ini")
        config.read(model_config_path)
        model_config = OrderedDict(config._sections["hyperparameters"].items())
        return tfrec_config, model_config

    def _create_logger(self, tfrec_config, model_config):
        # create logger
        data_name = tfrec_config["data_name"]
        model_name = tfrec_config["model"]
        log_dir = os.path.join("./log", data_name, model_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logger_name = '_'.join(["{}={}".format(arg, value) for arg, value in model_config.items()
                                if len(value) < 20])
        special_char = {'/', '\\', '\"', ':', '*', '?', '<', '>', '|', '\t'}
        logger_name = [c if c not in special_char else '_' for c in logger_name]
        logger_name = ''.join(logger_name)
        timestamp = time.time()
        # model name, param, timestamp
        logger_name = "%s_log_%s_%d.log" % (model_name, logger_name, timestamp)
        logger_name = os.path.join(log_dir, logger_name)
        logger = Logger(logger_name)

        # write configuration into log file
        info = '\n'.join(["{}={}".format(arg, value) for arg, value in tfrec_config.items()])
        logger.info("\nTFRec information:\n%s " % info)

        logger.info("\n")
        logger.info("Recommender:%s" % model_name)
        logger.info("Dataset name：\t%s" % data_name)
        argument = '\n'.join(["{}={}".format(arg, value) for arg, value in model_config.items()])
        logger.info("\nHyperparameters:\n%s " % argument)

        return logger

    def _eval_parameter(self, tfrec_config, model_config):
        # get parameters
        config = OrderedDict(tfrec_config, **model_config)
        for key, value in config.items():
            try:
                config[key] = eval(value)
            except:
                config[key] = value
        return config
