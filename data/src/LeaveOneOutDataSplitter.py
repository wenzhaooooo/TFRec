import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from utils.tools import random_choice
from data.src.Dataset import Dataset
from data.src.AbstractDataSplitter import AbstractDataSplitter


class LeaveOneOutDataSplitter(AbstractDataSplitter):
    def __init__(self, data_format='UIRT', sep=' ', user_min=3, item_min=None, negative_num=100):
        super(LeaveOneOutDataSplitter, self).__init__()
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
        if self.data_format == "UIRT":
            data = data.sort_values(by=["user", "time"])
        elif self.data_format == "UIR":
            data = data.sort_values(by=["user"])

        data = data.reset_index(drop=True)

        test_idx = data.groupby("user").size().cumsum() - 1
        valid_idx = test_idx - 1

        test_bool = np.zeros(ratings_num, dtype=np.bool)
        test_bool[test_idx] = True
        valid_bool = np.zeros(ratings_num, dtype=np.bool)
        valid_bool[valid_idx] = True
        train_bool = np.logical_not(np.logical_or(test_bool, test_bool))

        train_data = data[train_bool]
        valid_data = data[valid_bool]
        test_data = data[test_bool]

        train_matrix = csr_matrix((train_data["rating"], (train_data["user"], train_data["item"])),
                                  shape=(users_num, items_num))
        valid_matrix = csr_matrix((valid_data["rating"], (valid_data["user"], valid_data["item"])),
                                  shape=(users_num, items_num))
        test_matrix = csr_matrix((test_data["rating"], (test_data["user"], test_data["item"])),
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