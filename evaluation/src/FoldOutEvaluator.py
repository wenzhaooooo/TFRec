from evaluation.src.core.evaluate import evaluate_model
import numpy as np
from utils.tools import csr_to_user_dict, csr_to_user_item_pair
from evaluation.src.AbstractEvaluator import AbstractEvaluator
from data.DataLoader import get_data_loader
import sys


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
    def __init__(self, train_matrix, valid_matrix, test_matrix, test_negative=None):
        # TODO test the code correctness
        super(FoldOutEvaluator, self).__init__()
        if test_negative is not None:
            self.user_pos_train = None
            self.user_num = test_negative.shape[0]
            neg_users, neg_items = csr_to_user_item_pair(test_negative)
            self.negative_data = get_data_loader(neg_users, neg_items, batch_size=1024, shuffle=False)
            valid_users, valid_items = csr_to_user_item_pair(valid_matrix)
            self.valid_data = get_data_loader(valid_users, valid_items, batch_size=1024, shuffle=False)
            test_users, test_items = csr_to_user_item_pair(test_matrix)
            self.test_data = get_data_loader(test_users, test_items, batch_size=1024, shuffle=False)

            self.user_pos_valid, self.valid_nums = _csr_to_num(valid_matrix)  # reset valid items id
            self.valid_cumsum = np.zeros(self.user_num+1, dtype=np.intc)  # for reconstructing predicted rating matrix
            self.valid_cumsum[1:] = np.cumsum(self.valid_nums, dtype=np.intc)
            self.valid_max_num = np.max(self.valid_nums)  # for padding

            self.user_pos_test, self.test_nums = _csr_to_num(test_matrix)  # reset test items id
            self.test_cumsum = np.zeros(self.user_num+1, dtype=np.intc)  # for reconstructing predicted rating matrix
            self.test_cumsum[1:] = np.cumsum(self.test_nums, dtype=np.intc)
            self.test_max_num = np.max(self.test_nums)  # for padding
        else:
            self.user_pos_train = csr_to_user_dict(train_matrix)
            self.user_pos_valid = csr_to_user_dict(valid_matrix)
            self.user_pos_test = csr_to_user_dict(test_matrix)

    def print_metrics(self):
        """In NDCG, 'TOP' denotes that its idcg is calculated by the ranking of top-n items,
        'ALL' denotes that its idcg is calculated by the ranking of all positive items
        """
        print("Precision@5:5:50, Recall@5:5:50, MAP@5:5:50, NDCG_TOP@5:5:50, NDCG_ALL@5:5:50, MRR@5:5:50")

    def evaluate(self, model):
        if self.user_pos_train is not None:
            valid_result, test_result = self._evaluate_without_negative(model)
        else:
            valid_result, test_result = self._evaluate_with_negative(model)
        return valid_result, test_result

    def _evaluate_without_negative(self, model):
        ranking_score = model.get_ratings_matrix()
        valid_result = self._eval(ranking_score, self.user_pos_train, self.user_pos_valid)
        test_result = self._eval(ranking_score, self.user_pos_train, self.user_pos_test)
        return valid_result, test_result

    def _evaluate_with_negative(self, model):
        neg_ratings = []
        for users, items in self.negative_data:
            r_tmp = model.predict(users, items)
            neg_ratings.extend(r_tmp)
        neg_ratings = np.array(neg_ratings).reshape([self.user_num, -1])

        # calculate the predict ratings of valid items
        valid_ratings_flat = []
        for users, items in self.negative_data:
            r_tmp = model.predict(users, items)
            valid_ratings_flat.extend(r_tmp)

        # reconstruct and pad the predicted valid rating matrix
        valid_ratings = np.full([self.user_num, self.valid_max_num], -sys.float_info.max)
        for user, num in enumerate(self.valid_nums):
            low = self.valid_cumsum[user]
            high = self.valid_cumsum[user+1]
            valid_ratings[user][:num] = valid_ratings_flat[low:high]

        # calculate the predict ratings of test items
        test_ratings_flat = []
        for users, items in self.negative_data:
            r_tmp = model.predict(users, items)
            test_ratings_flat.extend(r_tmp)

        # reconstruct and pad the predicted test rating matrix
        test_ratings = np.full([self.user_num, self.test_max_num], -sys.float_info.max)
        for user, num in enumerate(self.test_nums):
            low = self.test_cumsum[user]
            high = self.test_cumsum[user + 1]
            test_ratings[user][:num] = test_ratings_flat[low:high]

        # put valid rating in front of negative rating,
        # because we reset valid items id from 0 to the number of test items
        rating_for_valid = np.hstack([valid_ratings, neg_ratings])
        # here, 'self.user_pos_train' is 'None'
        valid_result = self._eval(rating_for_valid, None, self.user_pos_valid)

        rating_for_test = np.hstack([test_ratings, neg_ratings])
        test_result = self._eval(rating_for_test, None, self.user_pos_test)
        return valid_result, test_result

    @staticmethod
    def _eval(ranking_score, user_pos_train, user_pos_test):
        result = evaluate_model(ranking_score, user_pos_train, user_pos_test)
        result = result[:, np.arange(4, 50, 5)]
        result = np.ndarray.flatten(result)
        return result
