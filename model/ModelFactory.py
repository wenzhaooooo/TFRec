from model.AOBPR import AOBPR
from model.BPR import BPR
from model.DNSBPR import DNSBPR
from model.APL import APL
from model.APL_ori import APL_ori
from model.IRGAN import IRGAN
from model.APL_Pro import APL_Pro
from model.AttAPL import AttAPL
from model.GANBPR import GANBPR


class ModelFactory(object):
    def __init__(self):
        self.model_dict = {"aobpr": AOBPR,
                           "bpr": BPR,
                           "dnsbpr": DNSBPR,
                           "apl": APL,
                           "irgan": IRGAN,
                           "apl_pro": APL_Pro,
                           "apl_ori": APL_ori,
                           "attapl": AttAPL,
                           "ganbpr": GANBPR
                           }

    def get_model(self, name):
        return self.model_dict[name.lower()]

    def available_model(self):
        return sorted(list(self.model_dict.keys()))
