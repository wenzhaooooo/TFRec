from model.AOBPR import AOBPR
from model.BPR import BPR
from model.DNSBPR import DNSBPR


class ModelFactory(object):
    def __init__(self):
        self.model_dict = {"aobpr": AOBPR,
                           "bpr": BPR,
                           "dnsbpr": DNSBPR}

    def create_model(self, name):
        return self.model_dict[name.lower()]

    def available_model(self):
        return sorted(list(self.model_dict.keys()))
