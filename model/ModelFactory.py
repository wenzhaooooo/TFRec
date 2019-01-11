from model.AOBPR import AOBPR
from model.BPR import BPR
from model.DNSBPR import DNSBPR
from model.BPRMF import BPRMF
#from model.FISM import FISM
#from model.GMF import GMF
#from model.MLP import MLP
#from model.NAIS import NAIS
#from model.NeuMF import NeuMF
#from model.NeuPR import NeuPR


class ModelFactory(object):
    def __init__(self):
        self.model_dict = {"aobpr": AOBPR,
                           "bpr": BPR,
                           "dnsbpr": DNSBPR,
                           #"fism": FISM,
                           "bprmf": BPRMF,
                           #"gmf": GMF,
                           #"mlp": MLP,
                           #"nais": NAIS,
                           #"neumf": NeuMF,
                           #"neupr": NeuPR
                           }

    def get_model(self, name):
        return self.model_dict[name.lower()]

    def available_model(self):
        return sorted(list(self.model_dict.keys()))
