from data.src.GivenRatioDataSplitter import GivenRatioDataSplitter
from data.src.LeaveOneOutDataSplitter import LeaveOneOutDataSplitter


class DataLoader(object):
    def __init__(self, data_info):
        # ratio, loo, test
        self.data_name = data_info["data_name"] if "data_name" in data_info else None

        self.data_file = data_info["data_file"] if "data_file" in data_info else None
        self.train_file = data_info["train_file"] if "train_file" in data_info else None
        self.valid_file = data_info["valid_file"] if "valid_file" in data_info else None
        self.test_file = data_info["test_file"] if "test_file" in data_info else None
        if self.data_file is not None:  # if data_file is not None, other files are invalid.
            self.train_file = self.data_file
            self.valid_file = None
            self.test_file = None

        self.format = data_info["format"] if "format" in data_info else None
        self.splitter = data_info["splitter"] if "splitter" in data_info else None
        self.separator = eval(data_info["separator"]) if "separator" in data_info else None

        self.user_min = eval(data_info["user_min"]) if "user_min" in data_info else None
        self.item_min = eval(data_info["item_min"]) if "item_min" in data_info else None

        self.train_ratio = eval(data_info["train_ratio"]) if "train_ratio" in data_info else None
        self.valid_ratio = eval(data_info["valid_ratio"]) if "valid_ratio" in data_info else None
        self.test_ratio = eval(data_info["test_ratio"]) if "test_ratio" in data_info else None
        self.evaluate_neg = eval(data_info["evaluate_neg"]) if "evaluate_neg" in data_info else None

    def load_data(self):
        # now, splitter just have two type, i.e., ratio and loo.
        # for each type, there are two case, need to split and already split.
        if self.splitter == "ratio":
            dataset = self._load_ratio()
        elif self.splitter == "loo":
            dataset = self._load_loo()
        else:
            raise ValueError("Please choose a valid splitter!")
        dataset.name = self.data_name
        return dataset

    def _load_ratio(self):
        splitter = GivenRatioDataSplitter(data_format=self.format, sep=self.separator,
                                          user_min=self.user_min, item_min=self.item_min,
                                          negative_num=self.evaluate_neg)
        return splitter.load_data(self.train_file, self.test_file, self.valid_file)

    def _load_loo(self):
        splitter = LeaveOneOutDataSplitter(data_format=self.format, sep=self.separator,
                                           user_min=self.user_min, item_min=self.item_min,
                                           negative_num=self.evaluate_neg)
        return splitter.load_data(self.train_file, self.test_file, self.valid_file)
