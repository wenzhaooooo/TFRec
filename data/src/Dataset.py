class Dataset(object):
    def __init__(self, train_matrix, valid_matrix, test_matrix, test_negative=None):
        self.train_matrix = train_matrix
        self.valid_matrix = valid_matrix
        self.test_matrix = test_matrix
        self.test_negative = test_negative
