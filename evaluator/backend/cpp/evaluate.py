try:
    from .apt_evaluate import apt_evaluate
except:
    raise ImportError("Import apt_evaluate error!")
from utils import typeassert
import numpy as np
import sys


@typeassert(rating_matrix=np.ndarray, user_pos_train=dict, user_pos_test=dict, top_k=int, thread_num=int)
def eval_rating_matrix(rating_matrix, user_pos_train, user_pos_test, top_k=50, thread_num=None):
    """
    :param rating_matrix: float32
    :param user_pos_train: type is 'dict', which key is user id, value is positive items list or array
    :param user_pos_test: type is 'dict', which key is user id, value is test items list or array
    :param top_k:
    :param thread_num:
    :return: Precision@5:5:50, Recall@5:5:50, MAP@5:5:50, NDCG_TOP@5:5:50, NDCG_ALL@5:5:50, MRR@5:5:50
            In NDCG, 'TOP' denotes that its idcg is calculated by the ranking of top-n items,
            'ALL' denotes that its idcg is calculated by the ranking of all positive items
    """
    if rating_matrix.dtype != np.float32:
        rating_matrix = np.array(rating_matrix, dtype=np.float32)

    if user_pos_train is not None:
        for user, items in user_pos_train.items():
            rating_matrix[user][items] = -sys.float_info.max

    test_items = []
    test_users = sorted(list(user_pos_test.keys()))
    for u in test_users:
        u_items = user_pos_test[u]
        if (not isinstance(u_items, np.ndarray)) or (u_items.dtype != np.intc) or (u_items.base is not None):
            u_items = np.array(u_items, dtype=np.intc, copy=True)
        test_items.append(u_items)

    if len(rating_matrix) != len(test_users):
        rating_matrix = rating_matrix[test_users]

    if rating_matrix.base is not None:
        rating_matrix = np.array(rating_matrix, dtype=np.intc, copy=True)

    results = apt_evaluate(rating_matrix, test_items, top_k, thread_num)
    return results
