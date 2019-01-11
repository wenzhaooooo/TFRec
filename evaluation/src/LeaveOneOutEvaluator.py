from evaluation.src.core.evaluate import evaluate_loo
from evaluation.src.AbstractEvaluator import AbstractEvaluator
import numpy as np
from utils.tools import csr_to_user_dict, csr_to_user_item_pair
from data.DataIterator import get_data_iterator
from evaluation.src.evaluate_leave_one_out import evaluate_leave_one_out


class LeaveOneOutEvaluator(AbstractEvaluator):
    """Evaluator for leave one item test set.
    valid_matrix, test_matrix and test_negative are the sparse matrix of scipy.sparse
    """
    def __init__(self, train_matrix, test_matrix, test_negative=None):
        # TODO add non-test_negative
        super(LeaveOneOutEvaluator, self).__init__()

        self.user_pos_train = None
        self.user_pos_test = None
        self.user_neg_test = None

        if train_matrix is not None:
            self.user_pos_train = csr_to_user_dict(train_matrix)
        if test_negative is not None:
            self.user_neg_test = csr_to_user_dict(test_negative)

        self.user_pos_test = csr_to_user_dict(test_matrix)

    def print_metrics(self):
        print("NDCG@5:5:50, HR@5:5:50")

    def evaluate(self, model):
        if self.user_neg_test is not None:
            result = evaluate_leave_one_out(model, None, self.user_pos_test, self.user_neg_test, top_k=50)
        else:
            result = evaluate_leave_one_out(model, self.user_pos_train, self.user_pos_test, None, top_k=50)
        result = result[:, np.arange(4, 50, 5)]
        return result.flatten()
