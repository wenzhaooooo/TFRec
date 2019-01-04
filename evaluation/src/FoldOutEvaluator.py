from evaluation.src.core.evaluate import evaluate_model
import numpy as np
from utils.tools import csr_to_dict
from evaluation.src.AbstractEvaluator import AbstractEvaluator


class FoldOutEvaluator(AbstractEvaluator):
    """Evaluator for generic ranking task.
    """
    def __init__(self, train_matrix, valid_matrix, test_matrix, test_negative=None):
        # TODO add test_negative
        super(FoldOutEvaluator, self).__init__()
        self.user_pos_train = csr_to_dict(train_matrix)
        self.user_pos_test = csr_to_dict(test_matrix)
        self.user_pos_valid = csr_to_dict(valid_matrix)

    def print_metrics(self):
        """In NDCG, 'TOP' denotes that its idcg is calculated by the ranking of top-n items,
        'ALL' denotes that its idcg is calculated by the ranking of all positive items
        """
        print("Precision@5:5:50, Recall@5:5:50, MAP@5:5:50, NDCG_TOP@5:5:50, NDCG_ALL@5:5:50, MRR@5:5:50")

    def evaluate(self, model):
        ranking_score = model.get_ratings_matrix()

        valid_result = evaluate_model(ranking_score, self.user_pos_train, self.user_pos_valid)
        valid_result = valid_result[:, np.arange(4, 50, 5)]
        valid_result = np.ndarray.flatten(valid_result)

        test_result = evaluate_model(ranking_score, self.user_pos_train, self.user_pos_test)
        test_result = test_result[:, np.arange(4, 50, 5)]
        test_result = np.ndarray.flatten(test_result)
        return valid_result, test_result
