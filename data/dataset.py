import numpy as np
from scipy.sparse import csr_matrix


class Dataset(object):
    def __init__(self, config):
        self.name = None
        self.train_file = None
        self.test_file = None

        self.num_users = None
        self.num_items = None
        self.num_ratings = None

        self.train_matrix = None
        self.test_matrix = None

        self._load_data(config)

    def _load_data(self, config):
        self.name = config["data_name"] if "data_name" in config else None
        self.train_file = config["train_file"] if "train_file" in config else None
        self.test_file = config["test_file"] if "test_file" in config else None

        file_format = config["format"] if "format" in config else None
        sep = config["separator"] if "separator" in config else None

        if file_format == "UIRT":
            columns = ["user", "item", "rating", "time"]
        elif file_format == "UIR":
            columns = ["user", "item", "rating"]
        else:
            raise ValueError("There is not data format '%s'" % file_format)

        train_data = np.genfromtxt(self.train_file, dtype=None, names=columns, delimiter=sep)
        test_data = np.genfromtxt(self.test_file, dtype=None, names=columns, delimiter=sep)

        all_data = np.concatenate([train_data, test_data])
        self.num_users = len(np.unique(all_data["user"]))
        self.num_items = len(np.unique(all_data["item"]))
        self.num_ratings = len(all_data)

        self.train_matrix = csr_matrix((train_data["rating"], (train_data["user"], train_data["item"])),
                                       shape=(self.num_users, self.num_items))
        self.test_matrix = csr_matrix((test_data["rating"], (test_data["user"], test_data["item"])),
                                      shape=(self.num_users, self.num_items))
