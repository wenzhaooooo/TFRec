from utils import typeassert, csr_to_user_dict
from scipy.sparse import csr_matrix
from .backend import eval_rating_matrix


class AbstractEvaluator(object):
    """Basic class for evaluator.
    """
    def __init__(self):
        pass

    def evaluate(self, ranking_score):
        raise NotImplementedError


class FoldOutEvaluator(AbstractEvaluator):
    """Evaluator for generic ranking task.
    """
    @typeassert(train_matrix=csr_matrix, test_matrix=csr_matrix, top_k=int)
    def __init__(self, train_matrix, test_matrix, top_k=50):
        super(FoldOutEvaluator, self).__init__()
        self.top_k = top_k
        self.user_pos_train = csr_to_user_dict(train_matrix)
        self.user_pos_test = csr_to_user_dict(test_matrix)

    def metrics_info(self):
        return "Precision@1:1:50, Recall@1:1:50, MAP@1:1:50, NDCG@1:1:50, MRR@1:1:50"

    def evaluate(self, model):
        ranking_score = model._predict()  # TODO rename
        result = eval_rating_matrix(ranking_score, self.user_pos_train, self.user_pos_test, top_k=self.top_k)
        return result.flatten()
