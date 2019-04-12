import numpy as np
from .metrics import precision, map, recall, ndcg, mrr
from utils import typeassert, argmax_top_k
from concurrent.futures import ThreadPoolExecutor


_rating_matrix = None
_user_pos_train = None
_user_pos_test = None
_top_k = None


@typeassert(rating_matrix=np.ndarray, user_pos_train=dict, user_pos_test=dict, top_k=int, thread_num=int)
def eval_rating_matrix(rating_matrix, user_pos_train, user_pos_test, top_k=50, thread_num=None):
    global _rating_matrix
    global _user_pos_train
    global _user_pos_test
    global _top_k

    _rating_matrix = rating_matrix
    _user_pos_train = user_pos_train
    _user_pos_test = user_pos_test
    _top_k = top_k

    test_user = list(user_pos_test.keys())
    test_num = len(test_user)

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        batch_result = executor.map(_evaluate_one_user, test_user)

    result = np.zeros(top_k * 5, dtype=np.float32)
    for re in batch_result:
        result += re
    ret = result / test_num
    ret = np.reshape(ret, [5, -1])
    return ret


def _evaluate_one_user(user):
    train_items = _user_pos_train[user]
    test_item = _user_pos_test[user]

    all_ratings = _rating_matrix[user]
    all_ratings = np.reshape(all_ratings, [-1])
    all_ratings[train_items] = -np.inf

    all_ranking = argmax_top_k(all_ratings, _top_k)

    result = []
    result.extend(precision(all_ranking, test_item))
    result.extend(recall(all_ranking, test_item))
    result.extend(map(all_ranking, test_item))
    result.extend(ndcg(all_ranking, test_item))
    result.extend(mrr(all_ranking, test_item))

    result = np.array(result, dtype=np.float32).flatten()
    return result
