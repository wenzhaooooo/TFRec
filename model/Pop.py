from model.base import AbstractRecommender
import numpy as np


class Pop(AbstractRecommender):
    def __init__(self, sess, config, dataset, evaluator):
        super(Pop, self).__init__(sess, config, dataset, evaluator)
        self.train_matrix = dataset.train_matrix
        self.users_num, self.items_num = self.train_matrix.shape
        self.evaluator = evaluator

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        item_user_matrix = self.train_matrix.transpose()
        self.ranking_score = np.array([items_line.nnz for items_line in item_user_matrix], dtype=np.float32)

        self.logger.info(self.evaluator.metrics_info())
        result = self.evaluate_model()
        self.logger.info("result:\t%s" % result)

    def _predict(self):
        ratings = np.reshape(np.tile(self.ranking_score, self.users_num),
                             newshape=[self.users_num, self.items_num])
        return ratings

    def evaluate_model(self):
        result = self.evaluator.evaluate(self)
        buf = '\t'.join([str(x) for x in result])
        return buf
