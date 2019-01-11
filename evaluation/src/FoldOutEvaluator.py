import numpy as np
from utils.tools import csr_to_user_dict
from evaluation.src.AbstractEvaluator import AbstractEvaluator
from evaluation.src.evaluate_fold_out import evaluate_fold_out


def _csr_to_num(csr_matrix_data):
    """used for resetting valid and test items id.
    return reset id dict and items num of each user.
    """
    user_num_dict = {}
    user_num = csr_matrix_data.shape[0]
    items_num_per_user = np.zeros(user_num, dtype=np.intc)
    for user, value in enumerate(csr_matrix_data):
        if any(value.indices):
            item_len = len(value.indices)
            user_num_dict[user] = np.arange(item_len)
            items_num_per_user[user] = item_len
    return user_num_dict, items_num_per_user


class FoldOutEvaluator(AbstractEvaluator):
    """Evaluator for generic ranking task.
    """
    def __init__(self, train_matrix, test_matrix, test_negative=None):
        # TODO test the code correctness
        super(FoldOutEvaluator, self).__init__()
        self.user_pos_train = None
        self.user_pos_test = None
        self.user_neg_test = None

        if train_matrix is not None:
            self.user_pos_train = csr_to_user_dict(train_matrix)
        if test_negative is not None:
            self.user_neg_test = csr_to_user_dict(test_negative)

        self.user_pos_test = csr_to_user_dict(test_matrix)

    def print_metrics(self):
        """In NDCG, 'TOP' denotes that its idcg is calculated by the ranking of top-n items,
        'ALL' denotes that its idcg is calculated by the ranking of all positive items
        """
        print("Precision@5:5:50, Recall@5:5:50, MAP@5:5:50, NDCG@5:5:50, MRR@5:5:50")

    def evaluate(self, model):
        if self.user_neg_test is not None:
            result = evaluate_fold_out(model, None, self.user_pos_test, self.user_neg_test, top_k=50)
        else:
            result = evaluate_fold_out(model, self.user_pos_train, self.user_pos_test, None, top_k=50)
        result = result[:, np.arange(4, 50, 5)]
        return result.flatten()
