import numpy as np
from utils.tools import argmax_top_k
from concurrent.futures import ThreadPoolExecutor
from evaluation.src.leave_one_out_metrics import ndcg, hr


_model = None
_user_pos_train = None
_user_pos_test = None
_user_neg_test = None
_top_k = None


def evaluate_leave_one_out(model,
                           user_pos_train=None,
                           user_pos_test=None,
                           user_neg_test=None,
                           top_k=50):

    if not (user_pos_train or user_neg_test):
        raise ValueError("lack evaluate information...")
    global _model
    global _user_pos_train
    global _user_pos_test
    global _user_neg_test
    global _top_k

    _model = model
    _user_pos_train = user_pos_train
    _user_pos_test = user_pos_test
    _user_neg_test = user_neg_test
    _top_k = top_k

    test_user = list(user_pos_test.keys())
    test_num = len(test_user)
    if _user_neg_test is not None:
        with ThreadPoolExecutor() as executor:
            batch_result = executor.map(_evaluate_one_user_with_negative, test_user)
    else:
        with ThreadPoolExecutor() as executor:
            batch_result = executor.map(_evaluate_one_user, test_user)

    result = np.zeros(top_k*2, dtype=np.float32)
    for re in batch_result:
        result += re
    ret = result / test_num
    ret = np.reshape(ret, [2, -1])
    return ret


def _evaluate_one_user_with_negative(user):
    test_item = _user_pos_test[user]
    neg_items = _user_neg_test[user]
    all_items = np.concatenate([[test_item], neg_items])
    test_item = 0

    all_ratings = _model.predict_for_eval(user, all_items)
    all_ranking = argmax_top_k(all_ratings, _top_k)

    result = []
    result.extend(ndcg(all_ranking, test_item))
    result.extend(hr(all_ranking, test_item))

    result = np.array(result, dtype=np.float32).flatten()
    return result


def _evaluate_one_user(user):
    train_items = _user_pos_train[user]
    test_item = _user_pos_test[user][0]

    all_rating = _model.predict_for_eval([user])
    all_rating = np.reshape(all_rating, [-1])

    all_rating[train_items] = -np.inf

    all_ranking = argmax_top_k(all_rating, _top_k)

    result = []
    result.extend(ndcg(all_ranking, test_item))
    result.extend(hr(all_ranking, test_item))

    result = np.array(result, dtype=np.float32).flatten()
    return result