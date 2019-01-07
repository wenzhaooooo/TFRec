import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from utils.tools import random_choice
from data.src.Dataset import Dataset
from data.src.AbstractDataSplitter import AbstractDataSplitter


class GivenRatioDataSplitter(AbstractDataSplitter):
    def __init__(self, data_format='UIRT', sep=' ', user_min=3, item_min=None, negative_num=None):
        super(GivenRatioDataSplitter, self).__init__()
        self.data_format = data_format
        self.sep = sep
        self.user_min = user_min
        self.item_min = item_min
        self.negative_num = negative_num

    def load_data(self, file_path):
        if self.data_format == "UIRT":
            columns = ["user", "item", "rating", "time"]
        elif self.data_format == "UIR":
            columns = ["user", "item", "rating"]
        else:
            raise ValueError("There is not data format '%s'" % self.data_format)

        data = pd.read_csv(file_path, names=columns, sep=self.sep, header=None)  # read file
        # filter users and items
        if self.user_min is not None and self.user_min > 0:
            user_cnt = data['user'].value_counts().to_dict()
            data = data[data['user'].map(lambda x: user_cnt[x] >= self.user_min)]
        if self.item_min is not None and self.item_min > 0:
            item_cnt = data['item'].value_counts().to_dict()
            data = data[data['item'].map(lambda x: item_cnt[x] >= self.item_min)]

        # statistic of dataset
        unique_users = np.sort(data["user"].unique())
        unique_items = np.sort(data["item"].unique())

        users_num = len(unique_users)
        items_num = len(unique_items)
        ratings_num = len(data)

        # remap users and items id
        user_remap = {}
        for i, user in enumerate(unique_users):
            user_remap[user] = i
        data['user'] = data['user'].map(lambda x: user_remap[x])
        item_remap = {}
        for i, item in enumerate(unique_items):
            item_remap[item] = i
        data['item'] = data['item'].map(lambda x: item_remap[x])

        # split data
        ratio = np.array([0.7, 0.1, 0.2])
        train_data = []
        valid_data = []
        test_data = []
        if self.data_format == "UIRT":
            data = data.sort_values(by=["user", "time"])
            user_grouped = data.groupby("user")
            for user, data_u in user_grouped:
                data_u = data_u.values
                num = len(data_u)
                sec = np.ceil(np.cumsum(ratio[:-1]*num)).astype(np.intc)
                train_tmp, valid_tmp, test_tmp = np.split(data_u, sec)
                train_data.extend(train_tmp)
                valid_data.extend(valid_tmp)
                test_data.extend(test_tmp)
        elif self.data_format == "UIR":
            # data = data.sort_values(by=["user"])
            user_grouped = data.groupby("user")
            for user, data_u in user_grouped:
                data_u = data_u.values
                num = len(data_u)
                np.random.shuffle(data_u)
                sec = np.ceil(np.cumsum(ratio[:-1]*num)).astype(np.intc)
                train_tmp, valid_tmp, test_tmp = np.split(data_u, sec)
                train_data.extend(train_tmp)
                valid_data.extend(valid_tmp)
                test_data.extend(test_tmp)

        train_data = np.array(train_data)
        valid_data = np.array(valid_data)
        test_data = np.array(test_data)

        train_matrix = csr_matrix(
            (train_data[:, 2], (train_data[:, 0].astype(np.intc), train_data[:, 1].astype(np.intc))),
            shape=(users_num, items_num))
        valid_matrix = csr_matrix(
            (valid_data[:, 2], (valid_data[:, 0].astype(np.intc), valid_data[:, 1].astype(np.intc))),
            shape=(users_num, items_num))
        test_matrix = csr_matrix(
            (test_data[:, 2], (test_data[:, 0].astype(np.intc), test_data[:, 1].astype(np.intc))),
            shape=(users_num, items_num))

        test_negative = None
        if self.negative_num and self.negative_num > 0:
            all_items = np.arange(items_num)
            test_negative = []
            for u in range(users_num):
                u_train_items = train_matrix.getrow(u).indices
                u_valid_items = valid_matrix.getrow(u).indices
                u_test_items = valid_matrix.getrow(u).indices
                exclusion = np.concatenate([u_train_items, u_valid_items, u_test_items])
                test_negative.append(random_choice(all_items, size=self.negative_num, exclusion=exclusion))

            indices = np.array(test_negative, dtype=np.intc).flatten()
            indptr = np.arange(0, len(indices) + 1, self.negative_num)
            n_flag = [1] * len(indices)
            test_negative = csr_matrix((n_flag, indices, indptr), shape=(users_num, items_num))

        return Dataset(train_matrix, valid_matrix, test_matrix, test_negative)