import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from utils.tools import random_choice


class AbstractDataSplitter(object):
    def __init__(self):
        pass

    def load_data(self, file_path):
        raise NotImplementedError


class Dataset(object):
    def __init__(self, train_matrix, valid_matrix, test_matrix, test_negative=None):
        self.train_matrix = train_matrix
        self.valid_matrix = valid_matrix
        self.test_matrix = test_matrix
        self.test_negative = test_negative


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
            #data = data.sort_values(by=["user"])
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


class GivenTestSetDataSplitter(AbstractDataSplitter):
    def __init__(self, data_format='UIRT', sep=' ', user_min=None, item_min=3, negative_num=None):
        super(GivenTestSetDataSplitter, self).__init__()
        self.data_format = data_format
        self.sep = sep
        self.user_min = user_min
        self.item_min = item_min
        self.negative_num = negative_num

    def load_data(self, train_file, test_file, valid_file=None):
        if self.data_format == "UIRT":
            columns = ["user", "item", "rating", "time"]
        elif self.data_format == "UIR":
            columns = ["user", "item", "rating"]
        else:
            raise ValueError("There is not data format '%s'" % self.data_format)

        train_data = pd.read_csv(train_file, names=columns, sep=self.sep, header=None)  # read file
        test_data = pd.read_csv(test_file, names=columns, sep=self.sep, header=None)  # read file
        if valid_file is not None:
            valid_data = pd.read_csv(valid_file, names=columns, sep=self.sep, header=None)
        else:
            train_list, valid_list = [], []
            if self.data_format == "UIRT":
                # train_data = train_data.sort_values(by=["user", "time"])
                user_grouped = train_data.groupby("user")
                for user, data_u in user_grouped:
                    data_u = data_u.values
                    num = len(data_u)
                    sec = np.ceil([7.0/8*num])
                    train_tmp, valid_tmp = np.split(data_u, sec)
                    train_list.extend(train_tmp)
                    valid_list.extend(valid_tmp)
            elif self.data_format == "UIR":
                # train_data = train_data.sort_values(by=["user", "item"])
                user_grouped = train_data.groupby("user")
                for user, data_u in user_grouped:
                    data_u = data_u.values
                    num = len(data_u)
                    np.random.shuffle(data_u)
                    sec = np.ceil([7.0/8*num])
                    train_tmp, valid_tmp = np.split(data_u, sec)
                    train_list.extend(train_tmp)
                    valid_list.extend(valid_tmp)

            train_data = pd.DataFrame(train_list, columns=columns)
            valid_data = pd.DataFrame(valid_list, columns=columns)

        all_data = pd.concat([train_data, valid_data, test_data])

        # statistic of dataset
        unique_users = np.sort(all_data["user"].unique())
        unique_items = np.sort(all_data["item"].unique())

        users_num = len(unique_users)
        items_num = len(unique_items)

        # remap users and items id
        user_remap = {}
        for i, user in enumerate(unique_users):
            user_remap[user] = i
        train_data['user'] = train_data['user'].map(lambda x: user_remap[x])
        valid_data['user'] = valid_data['user'].map(lambda x: user_remap[x])
        test_data['user'] = test_data['user'].map(lambda x: user_remap[x])

        item_remap = {}
        for i, item in enumerate(unique_items):
            item_remap[item] = i
        train_data['item'] = train_data['item'].map(lambda x: item_remap[x])
        valid_data['item'] = valid_data['item'].map(lambda x: item_remap[x])
        test_data['item'] = test_data['item'].map(lambda x: item_remap[x])

        # convert to sparse matrix
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


if __name__ == "__main__":
    loo_splitter = GivenRatioDataSplitter(sep="::", negative_num=100)
    data = loo_splitter.load_data("/home/sun/Desktop/SunLib/datasets/ratings.dat")
    print()