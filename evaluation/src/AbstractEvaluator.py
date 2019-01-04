class AbstractEvaluator(object):
    """Basic class for evaluator.
    """
    def __init__(self):
        pass

    def print_metrics(self):
        raise NotImplementedError

    def evaluate(self, ranking_score):
        raise NotImplementedError
