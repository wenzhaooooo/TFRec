import numpy as np
from scipy.sparse import csr_matrix
from data.src.AbstractDataSplitter import AbstractDataSplitter
from data.Dataset import Dataset
from data.tools import filter_data, remap_id, sampling_negative, split_data_by_loo


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

        data = np.genfromtxt(file_path, dtype=None, names=columns, delimiter=self.sep)
        # filter data
        data = filter_data(data, user_min=self.user_min, item_min=self.item_min)
        # remap user and item id and return the remapped information
        data, user2id, item2id, id2user, id2item = remap_id(data)
        # store dataset information
        dataset = Dataset()
        dataset.user2id, dataset.item2id, dataset.id2user, dataset.id2item = user2id, item2id, id2user, id2item
        dataset.num_users, dataset.num_items = len(user2id), len(item2id)
        dataset.num_ratings = len(data)

        # split data
        if self.data_format == "UIRT":
            by_time = True
        elif self.data_format == "UIR":
            by_time = False

        train_data, test_data = split_data_by_loo(data, by_time=by_time)
        train_data, valid_data = split_data_by_loo(train_data, by_time=by_time)

        # construct sparse matrix
        train_matrix = csr_matrix((train_data["rating"], (train_data["user"], train_data["item"])),
                                  shape=(dataset.num_users, dataset.num_items))
        valid_matrix = csr_matrix((valid_data["rating"], (valid_data["user"], valid_data["item"])),
                                  shape=(dataset.num_users, dataset.num_items))
        test_matrix = csr_matrix((test_data["rating"], (test_data["user"], test_data["item"])),
                                 shape=(dataset.num_users, dataset.num_items))
        dataset.train_matrix, dataset.valid_matrix, dataset.test_matrix = \
            train_matrix, valid_matrix, test_matrix

        # sampling negative items for test
        test_negative = None
        if self.negative_num and self.negative_num > 0:
            test_negative = sampling_negative(train_matrix, valid_matrix,
                                              test_matrix, neg_num=self.negative_num)
        dataset.test_negative = test_negative

        return dataset
