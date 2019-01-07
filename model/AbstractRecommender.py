import configparser


class AbstractRecommender(object):
    def __init__(self):
        config = configparser.ConfigParser()
        config.read("conf/%s.ini" % self.__class__.__name__)
        self.conf = dict(config.items("hyperparameters"))
        print("%s arguments: %s " % (self.__class__.__name__, self.conf))

    def build_model(self):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError
    
    def evaluate_model(self): 
        raise NotImplementedError  