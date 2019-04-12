import numpy as np
from utils import typeassert
from collections import defaultdict, OrderedDict


@typeassert(filename=str, sep=str, columns=list)
def load_data(filename, sep, columns):
    data = np.genfromtxt(filename, dtype=None, names=columns, delimiter=sep)
    return data


@typeassert(data=np.ndarray)
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


@typeassert(data=np.ndarray)
def remap_id(data):
    unique_user = np.unique(data["user"])
    user2id = OrderedDict([(user, id) for id, user in enumerate(unique_user)])
    vfunc = np.vectorize(lambda x: user2id[x])
    data["user"] = vfunc(data["user"])

    unique_item = np.unique(data["item"])
    item2id = OrderedDict([(item, id) for id, item in enumerate(unique_item)])
    vfunc = np.vectorize(lambda x: item2id[x])
    data["item"] = vfunc(data["item"])

    return data, user2id, item2id


@typeassert(data=np.ndarray, ratio=float, by_time=bool)
def split_by_ratio(data, ratio=0.8, by_time=True):
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
        sec = np.ceil([ratio*len_lines]).astype(np.intc)
        first_tmp, second_tmp = np.split(lines, sec)
        first_section.extend(first_tmp)
        second_section.extend(second_tmp)

    first_section = np.array(first_section, dtype=None)
    second_section = np.array(second_section, dtype=None)
    return first_section, second_section


@typeassert(data=np.ndarray, by_time=bool)
def split_by_loo(data, by_time=True):
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
            first_section.extend(lines)
        else:
            if not by_time:
                np.random.shuffle(lines)
            first_section.extend(lines[:-1])
            second_section.append(lines[-1])
    first_section = np.array(first_section, dtype=None)
    second_section = np.array(second_section, dtype=None)
    return first_section, second_section
