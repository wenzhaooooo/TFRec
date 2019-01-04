from evaluation.src.core.evaluate import evaluate_loo
from evaluation.src.AbstractEvaluator import AbstractEvaluator
import numpy as np
from data.DataLoader import get_data_loader


class LeaveOneOutEvaluator(AbstractEvaluator):
    """Evaluator for leave one item test set.
    valid_matrix, test_matrix and test_negative are the sparse matrix of scipy.sparse
    """
    def __init__(self, train_matrix, valid_matrix, test_matrix, test_negative=None):
        # TODO add non-test_negative
        super(LeaveOneOutEvaluator, self).__init__()
        self.user_num = test_negative.shape[0]
        self.negative_num = test_negative.getrow(0).nnz
        users, neg_items, valid_item, test_item = [], [], [], []

        for u in range(self.user_num):
            n_i = test_negative.getrow(u).indices
            users.extend([u]*len(n_i))
            neg_items.extend(n_i)
            valid_item.extend(valid_matrix.getrow(u).indices)
            test_item.extend(test_matrix.getrow(u).indices)

        self.negative_data = get_data_loader(users, neg_items, batch_size=1024, shuffle=False)
        self.valid_data = get_data_loader(np.arange(self.user_num), valid_item, batch_size=1024, shuffle=False)
        self.test_data = get_data_loader(np.arange(self.user_num), test_item, batch_size=1024, shuffle=False)

        self.test_item = np.full(self.user_num, self.negative_num)

    def print_metrics(self):
        print("NDCG_TOP@5:5:50, HR@5:5:50")

    def evaluate(self, model):
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
        valid_ratings = np.array(np.hstack([negative_ratings, valid_ratings]), dtype=np.float32, copy=True)

        test_ratings = []
        for users, items in self.test_data:
            r_tmp = model.predict(users, items)
            test_ratings.extend(r_tmp)
        test_ratings = np.array(test_ratings, dtype=np.float32)
        test_ratings = np.reshape(test_ratings, newshape=[self.user_num, -1])
        test_ratings = np.array(np.hstack([negative_ratings, test_ratings]), copy=True)

        valid_result = evaluate_loo(valid_ratings, self.test_item)
        valid_result = valid_result[:, np.arange(4, 50, 5)]
        valid_result = np.ndarray.flatten(valid_result)

        test_result = evaluate_loo(test_ratings, self.test_item)
        test_result = test_result[:, np.arange(4, 50, 5)]
        test_result = np.ndarray.flatten(test_result)

        return valid_result, test_result
