class AbstractRecommender(object):
    logger = None

    def __init__(self):
        pass

    def train_model(self):
        raise NotImplementedError
    
    def evaluate_model(self, epoch):
        raise NotImplementedError

    def _predict(self):
        """This function is used to evaluate model performance
        :return: all users' rating matrix
        """
        raise NotImplementedError
