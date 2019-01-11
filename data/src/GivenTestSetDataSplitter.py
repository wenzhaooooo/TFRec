import numpy as np
from scipy.sparse import csr_matrix
from data.src.AbstractDataSplitter import AbstractDataSplitter
from data.tools import remap_id, sampling_negative, split_data_by_ratio
from data.Dataset import Dataset


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

        # read data
        train_data = np.genfromtxt(train_file, dtype=None, names=columns, delimiter=self.sep)
        test_data = np.genfromtxt(test_file, dtype=None, names=columns, delimiter=self.sep)
        if valid_file is not None:
            valid_data = np.genfromtxt(valid_file, dtype=None, names=columns)
        else:
            if self.data_format == "UIRT":
                by_time = True
            elif self.data_format == "UIR":
                by_time = False
            train_data, valid_data = split_data_by_ratio(train_data, section=0.7/0.8, by_time=by_time)

        all_data = np.vstack([train_data, valid_data, test_data])
        all_data, user2id, item2id, id2user, id2item = remap_id(all_data)

        # store dataset information
        dataset = Dataset()
        dataset.user2id, dataset.item2id, dataset.id2user, dataset.id2item = user2id, item2id, id2user, id2item
        dataset.num_users, dataset.num_items = len(user2id), len(item2id)
        dataset.num_ratings = len(all_data)

        # remap user id
        vfunc = np.vectorize(lambda x: user2id[x])
        train_data["user"] = vfunc(train_data["user"])
        test_data["user"] = vfunc(test_data["user"])
        valid_data["user"] = vfunc(valid_data["user"])

        # remap item id
        vfunc = np.vectorize(lambda x: item2id[x])
        train_data["item"] = vfunc(train_data["item"])
        test_data["item"] = vfunc(test_data["item"])
        valid_data["item"] = vfunc(valid_data["item"])

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
