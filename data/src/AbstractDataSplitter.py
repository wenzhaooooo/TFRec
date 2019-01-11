class AbstractDataSplitter(object):
    def __init__(self):
        pass

    def load_data(self, train_file, test_file=None, valid_file=None):
        raise NotImplementedError
