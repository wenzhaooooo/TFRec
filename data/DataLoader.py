from data.src.GivenTestSetDataSplitter import GivenTestSetDataSplitter
from data.src.GivenRatioDataSplitter import GivenRatioDataSplitter
from data.src.LeaveOneOutDataSplitter import LeaveOneOutDataSplitter


class DataLoader(object):
    def __init__(self, path, splitter, separator, evaluate_neg, dataset_name):
        # ratio, loo, test
        self.path = path
        self.splitter = splitter
        self.separator = separator
        self.evaluate_neg = evaluate_neg
        self.dataset_name = dataset_name

    def load_data(self):
        if self.splitter == "test":
            dataset = self._load_test(self.path, self.separator, self.evaluate_neg)
        elif self.splitter == "ratio":
            dataset = self._load_ratio(self.path, self.separator, self.evaluate_neg)
        elif self.splitter == "loo":
            dataset = self._load_loo(self.path, self.separator, self.evaluate_neg)
        else:
            raise ValueError("Please choose a valid splitter!")
        dataset.name = self.dataset_name
        return dataset

    @staticmethod
    def _load_test(path, separator, evaluate_neg):
        splitter = GivenTestSetDataSplitter(data_format='UIRT', sep=separator, negative_num=evaluate_neg)
        return splitter.load_data(path + ".train.rating", path + ".test.rating", path + ".valid.rating")

    @staticmethod
    def _load_ratio(path, separator, evaluate_neg):
        splitter = GivenRatioDataSplitter(data_format='UIRT', sep=separator, negative_num=evaluate_neg)
        return splitter.load_data(path + ".dat")

    @staticmethod
    def _load_loo(path, separator, evaluate_neg):
        splitter = LeaveOneOutDataSplitter(data_format='UIRT', sep=separator, negative_num=evaluate_neg)
        return splitter.load_data(path + ".dat")
