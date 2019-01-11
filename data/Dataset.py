class Dataset(object):
    def __init__(self):
        self.name = None
        self.train_matrix = None
        self.valid_matrix = None
        self.test_matrix = None
        self.test_negative = None

        self.num_users = None
        self.num_items = None
        self.num_ratings = None

        self.user2id = None
        self.id2user = None
        self.item2id = None
        self.id2item = None
