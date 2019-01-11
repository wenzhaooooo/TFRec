import numpy as np
from utils.tools import random_choice
from scipy.sparse import csr_matrix
from collections import defaultdict


def filter_data(data, user_min=None, item_min=None):
    if item_min is not None and item_min > 0:
        items = data["item"]
        unique_item, counts = np.unique(items, return_counts=True)
        item_count = {item: count for item, count in zip(unique_item, counts)}
        filtered_idx = [item_count[i] >= item_min for i in items]
        data = data[filtered_idx]

    if user_min is not None and user_min > 0:
        users = data["user"]
        unique_user, counts = np.unique(users, return_counts=True)
        user_count = {user: count for user, count in zip(unique_user, counts)}
        filtered_idx = [user_count[u] >= user_min for u in users]
        data = data[filtered_idx]
    return data


def remap_id(data):
    unique_user = np.unique(data["user"])
    user2id = {user: id for id, user in enumerate(unique_user)}
    id2user = {id: user for id, user in enumerate(unique_user)}
    vfunc = np.vectorize(lambda x: user2id[x])
    data["user"] = vfunc(data["user"])

    unique_item = np.unique(data["item"])
    item2id = {item: id for id, item in enumerate(unique_item)}
    id2item = {id: item for id, item in enumerate(unique_item)}
    vfunc = np.vectorize(lambda x: item2id[x])
    data["item"] = vfunc(data["item"])

    return data, user2id, item2id, id2user, id2item


def sampling_negative(train_matrix, valid_matrix, test_matrix, neg_num=100):
    if neg_num is not None and neg_num > 0:
        users_num, items_num = train_matrix.shape
        all_items = np.arange(items_num)
        test_negative = []
        for u in range(users_num):
            u_train_items = train_matrix.getrow(u).indices
            u_valid_items = valid_matrix.getrow(u).indices
            u_test_items = test_matrix.getrow(u).indices
            exclusion = np.concatenate([u_train_items, u_valid_items, u_test_items])
            test_negative.append(random_choice(all_items, size=neg_num, exclusion=exclusion))

        indices = np.array(test_negative, dtype=np.intc).flatten()
        indptr = np.arange(0, len(indices) + 1, neg_num)
        n_flag = [1] * len(indices)
        test_negative = csr_matrix((n_flag, indices, indptr), shape=(users_num, items_num))
    else:
        test_negative = None
    return test_negative


def split_data_by_loo(data, by_time=True):
    if by_time:
        data.sort(order=["user", "time"])
    else:
        data.sort(order=["user", "item"])

    user_dict = defaultdict(list)
    for line in data:
        user_dict[line["user"]].append(line)

    first_section = []
    second_section = []
    for user, lines in user_dict.items():
        lines = np.array(lines, dtype=None)
        len_lines = len(lines)
        if len_lines < 3:
            # TODO if len_lines==1?
            first_section.extend(lines)
        else:
            if not by_time:
                np.random.shuffle(lines)
            first_section.extend(lines[:-1])
            second_section.append(lines[-1])
    first_section = np.array(first_section, dtype=None)
    second_section = np.array(second_section, dtype=None)
    return first_section, second_section


def split_data_by_ratio(data, section, by_time=True):
    if by_time:
        data.sort(order=["user", "time"])
    else:
        data.sort(order=["user", "item"])

    user_dict = defaultdict(list)
    for line in data:
        user_dict[line["user"]].append(line)

    first_section = []
    second_section = []
    for user, lines in user_dict.items():
        lines = np.array(lines, dtype=None)
        len_lines = len(lines)
        if not by_time:
            np.random.shuffle(lines)
        sec = np.ceil([section*len_lines]).astype(np.intc)
        first_tmp, second_tmp = np.split(lines, sec)
        first_section.extend(first_tmp)
        second_section.extend(second_tmp)

    first_section = np.array(first_section, dtype=None)
    second_section = np.array(second_section, dtype=None)
    return first_section, second_section
