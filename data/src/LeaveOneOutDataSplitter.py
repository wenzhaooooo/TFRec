import numpy as np
from scipy.sparse import csr_matrix
from data.src.AbstractDataSplitter import AbstractDataSplitter
from data.Dataset import Dataset
from data.tools import filter_data, remap_id, sampling_negative, split_data_by_loo


class LeaveOneOutDataSplitter(AbstractDataSplitter):
    def __init__(self, data_format='UIRT', sep=' ', user_min=None, item_min=None, negative_num=100, is_remap_id=False):
        super(LeaveOneOutDataSplitter, self).__init__()
        self.data_format = data_format
        self.sep = sep
        self.user_min = user_min
        self.item_min = item_min
        self.is_remap_id = is_remap_id

        self.negative_num = negative_num

    def load_data(self, train_file, valid_file=None, test_file=None):
        if self.data_format == "UIRT":
            columns = ["user", "item", "rating", "time"]
            by_time = True
        elif self.data_format == "UIR":
            columns = ["user", "item", "rating"]
            by_time = False
        else:
            raise ValueError("There is not data format '%s'" % self.data_format)

        if valid_file is not None:  # all the train, valid and test data are already split
            train_data = np.genfromtxt(train_file, dtype=None, names=columns, delimiter=self.sep)
            valid_data = np.genfromtxt(valid_file, dtype=None, names=columns, delimiter=self.sep)
            test_data = np.genfromtxt(test_file, dtype=None, names=columns, delimiter=self.sep)
        else:  # data not split at all, split train, valid and test set
            data = np.genfromtxt(train_file, dtype=None, names=columns, delimiter=self.sep)
            data = filter_data(data, user_min=self.user_min, item_min=self.item_min)  # filter data
            train_data, test_data = split_data_by_loo(data, by_time=by_time)
            train_data, valid_data = split_data_by_loo(train_data, by_time=by_time)

        all_data = np.concatenate([train_data, valid_data, test_data])
        dataset = Dataset()
        dataset.num_users, dataset.num_items = len(np.unique(all_data["user"])), len(np.unique(all_data["item"]))
        dataset.num_ratings = len(all_data)
        if self.is_remap_id:
            all_data, user2id, item2id, id2user, id2item = remap_id(all_data)  # get user and item id remap information

            # store remap and dataset information
            dataset = Dataset()
            dataset.user2id, dataset.item2id, dataset.id2user, dataset.id2item = user2id, item2id, id2user, id2item

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
