class AbstractDataSplitter(object):
    def __init__(self):
        pass

    def load_data(self, file_path):
        raise NotImplementedError
