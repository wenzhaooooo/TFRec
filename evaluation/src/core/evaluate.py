from .apt_evaluate import apt_evaluate
from .apt_evaluate_loo import apt_evaluate_loo
import numpy as np
import sys


def evaluate_model(all_ratings, user_train_dict, user_pos_test, top_k=50, thread_num=None):
    """
    :param all_ratings: float32
    :param user_train_dict: type is 'dict', which key is user id, value is positive items list or array
    :param user_pos_test: type is 'dict', which key is user id, value is test items list or array
    :param top_k:
    :param thread_num:
    :return: Precision@5:5:50, Recall@5:5:50, MAP@5:5:50, NDCG_TOP@5:5:50, NDCG_ALL@5:5:50, MRR@5:5:50
            In NDCG, 'TOP' denotes that its idcg is calculated by the ranking of top-n items,
            'ALL' denotes that its idcg is calculated by the ranking of all positive items
    """
    if (not isinstance(all_ratings, np.ndarray)) or all_ratings.dtype != np.float32:
        all_ratings = np.array(all_ratings, dtype=np.float32)

    if user_train_dict is not None:
        for user, items in user_train_dict.items():
            all_ratings[user][items] = -sys.float_info.max

    test_items = []
    test_users = sorted(list(user_pos_test.keys()))
    for u in test_users:
        u_items = user_pos_test[u]
        if (not isinstance(u_items, np.ndarray)) or (u_items.dtype != np.intc) or (u_items.base is not None):
            u_items = np.array(u_items, dtype=np.intc, copy=True)
        test_items.append(u_items)

    if len(all_ratings) != len(test_users):
        all_ratings = all_ratings[test_users]

    if all_ratings.base is not None:
        all_ratings = np.array(all_ratings, dtype=np.intc, copy=True)

    results = apt_evaluate(all_ratings, test_items, top_k, thread_num)
    return results


def evaluate_loo(all_ratings, user_train_dict, test_items, top_k=50, thread_num=None):
    # ensure the type of data
    if (not isinstance(all_ratings, np.ndarray)) \
            or (all_ratings.dtype != np.float32) \
            or (all_ratings.base is not None):
        all_ratings = np.array(all_ratings, dtype=np.float32, copy=True)

    # TODO if the length of 'all_ratings' has changed, the index is not consist with user id.
    if user_train_dict is not None:
        for user, items in user_train_dict.items():
            all_ratings[user][items] = -sys.float_info.max

    if (not isinstance(test_items, np.ndarray))\
            or (test_items.dtype != np.intc) \
            or (test_items.base is not None):
        test_items = np.array(test_items, dtype=np.intc, copy=True)

    results = apt_evaluate_loo(all_ratings, test_items, top_k, thread_num)
    return results
