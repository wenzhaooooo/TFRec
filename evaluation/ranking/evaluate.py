from .core.apt_evaluate import apt_evaluate
from .core.apt_evaluate_loo import apt_evaluate_loo
import numpy as np
import sys


def evaluate_model(all_ratings, user_pos_train, user_pos_test, top_k=None, rank_len=50, thread_num=None):
    """
    :param all_ratings: float32
    :param user_pos_train: type is 'dict', which key is user id, value is positive items list or array
    :param user_pos_test: type is 'dict', which key is user id, value is test items list or array
    :param rank_len:
    :param top_k:
    :param thread_num:
    :return: Precision@5:5:50, Recall@5:5:50, MAP@5:5:50, NDCG_TOP@5:5:50, NDCG_ALL@5:5:50, MRR@5:5:50
            In NDCG, 'TOP' denotes that its idcg is calculated by the ranking of top-n items,
            'ALL' denotes that its idcg is calculated by the ranking of all positive items
    """
    if isinstance(all_ratings, np.ndarray) and all_ratings.dtype != np.float32:
        all_ratings = np.array(all_ratings, dtype=np.float32)
    top_k = np.array(top_k) if top_k else np.arange(5, rank_len + 1, 5)

    for u in user_pos_test:
        all_ratings[u][user_pos_train[u]] = -sys.float_info.max

    test_items = []
    test_users = list(user_pos_test.keys())
    for u in test_users:
        test_items.append(user_pos_test[u])

    all_ratings = np.array(all_ratings[test_users], dtype=np.float32)

    results = apt_evaluate(all_ratings, test_items, rank_len, thread_num)
    metrics_value = results#[:, top_k-1]
    return metrics_value.flatten()


def evaluate_loo(all_ratings, test_items, top_k=50, thread_num=None):
    if isinstance(all_ratings, np.ndarray) and all_ratings.dtype != np.float32:
        all_ratings = np.array(all_ratings, dtype=np.float32)
    if isinstance(test_items, np.ndarray) and test_items.dtype != np.intc:
        test_items = np.array(test_items, dtype=np.intc)

    results = apt_evaluate_loo(all_ratings, test_items, top_k, thread_num)
    metrics_value = results
    return metrics_value.flatten()
