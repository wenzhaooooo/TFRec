from model.AOBPR import AOBPR
from model.BPR import BPR
from model.DNSBPR import DNSBPR
from model.APL import APL
from model.APL_ori import APL_ori


class ModelFactory(object):
    def __init__(self):
        self.model_dict = {"aobpr": AOBPR,
                           "bpr": BPR,
                           "dnsbpr": DNSBPR,
                           "apl": APL,
                           "apl_ori": APL_ori
                           }

    def get_model(self, name):
        return self.model_dict[name.lower()]

    def available_model(self):
        return sorted(list(self.model_dict.keys()))
