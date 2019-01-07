from evaluation.src.core.evaluate import evaluate_loo
from evaluation.src.AbstractEvaluator import AbstractEvaluator
import numpy as np
from utils.tools import csr_to_user_dict, csr_to_user_item_pair
from data.DataLoader import get_data_loader


class LeaveOneOutEvaluator(AbstractEvaluator):
    """Evaluator for leave one item test set.
    valid_matrix, test_matrix and test_negative are the sparse matrix of scipy.sparse
    """
    def __init__(self, train_matrix, valid_matrix, test_matrix, test_negative=None):
        # TODO add non-test_negative
        super(LeaveOneOutEvaluator, self).__init__()
        if valid_matrix.shape[0] != valid_matrix.nnz:
            # TODO valid item number os not equal to user number
            raise ValueError("valid item number os not equal to user number")
        if test_matrix.shape[0] != test_matrix.nnz:
            # TODO test item number os not equal to user number
            raise ValueError("test item number os not equal to user number")

        if test_negative is not None:
            self.user_pos_train = None
            self.user_num = test_negative.shape[0]
            neg_users, neg_items = csr_to_user_item_pair(test_negative)
            self.negative_data = get_data_loader(neg_users, neg_items, batch_size=1024, shuffle=False)
            valid_users, valid_items = csr_to_user_item_pair(valid_matrix)
            self.valid_data = get_data_loader(valid_users, valid_items, batch_size=1024, shuffle=False)
            test_users, test_items = csr_to_user_item_pair(test_matrix)
            self.test_data = get_data_loader(test_users, test_items, batch_size=1024, shuffle=False)

            self.test_item = np.zeros([self.user_num, 1], dtype=np.intc)
        else:
            self.user_pos_train = csr_to_user_dict(train_matrix)
            _, self.valid_item = csr_to_user_item_pair(valid_matrix)
            _, self.test_item = csr_to_user_item_pair(test_matrix)

    def print_metrics(self):
        print("NDCG_TOP@5:5:50, HR@5:5:50")

    def evaluate(self, model):
        if self.user_pos_train is not None:
            valid_result, test_result = self._evaluate_without_negative(model)
        else:
            valid_result, test_result = self._evaluate_with_negative(model)
        return valid_result, test_result

    def _evaluate_without_negative(self, model):
        ranking_score = model.get_ratings_matrix()
        valid_result = self._eval(ranking_score, self.user_pos_train, self.valid_item)
        test_result = self._eval(ranking_score, self.user_pos_train, self.test_item)
        return valid_result, test_result

    def _evaluate_with_negative(self, model):
        negative_ratings = []
        for users, items in self.negative_data:
            r_tmp = model.predict(users, items)
            negative_ratings.extend(r_tmp)

        negative_ratings = np.array(negative_ratings, dtype=np.float32)
        negative_ratings = np.reshape(negative_ratings, newshape=[self.user_num, -1])

        valid_ratings = []
        for users, items in self.valid_data:
            r_tmp = model.predict(users, items)
            valid_ratings.extend(r_tmp)
        valid_ratings = np.array(valid_ratings, dtype=np.float32)
        valid_ratings = np.reshape(valid_ratings, newshape=[self.user_num, -1])
        valid_ratings = np.hstack([valid_ratings, negative_ratings])

        test_ratings = []
        for users, items in self.test_data:
            r_tmp = model.predict(users, items)
            test_ratings.extend(r_tmp)
        test_ratings = np.array(test_ratings, dtype=np.float32)
        test_ratings = np.reshape(test_ratings, newshape=[self.user_num, -1])
        test_ratings = np.hstack([test_ratings, negative_ratings])

        valid_result = self._eval(valid_ratings, None, self.test_item)
        test_result = self._eval(test_ratings, None, self.test_item)

        return valid_result, test_result

    @staticmethod
    def _eval(ranking_score, user_pos_train, test_item):
        result = evaluate_loo(ranking_score, user_pos_train, test_item)
        result = result[:, np.arange(4, 50, 5)]
        result = np.ndarray.flatten(result)
        return result
