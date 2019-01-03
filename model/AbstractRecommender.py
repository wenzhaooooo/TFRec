class AbstractRecommender(object):
    def __init__(self):  
        pass

    def build_model(self):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError
    
    def evaluate_model(self): 
        raise NotImplementedError  