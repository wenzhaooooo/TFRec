class AbstractDataSplitter(object):
    def __init__(self):
        pass

    def load_data(self, train_file, valid_file=None, test_file=None):
        raise NotImplementedError
