import numpy as np
from utils.tools import random_choice
from data.DataIterator import get_data_iterator
from model.BPR import BPR


class AOBPR(BPR):
    def get_training_data(self):
        users = []
        pos_items = []
        neg_items = []
        for u, pos in self.user_pos_train.items():
            pos_len = len(pos)
            feed = {self.user_h: [u]}
            logits = self.sess.run(self.all_logits, feed_dict=feed)

            neg_exp = np.exp(logits)
            neg_prob = neg_exp / np.sum(neg_exp)
            neg = random_choice(self.all_items, size=pos_len, exclusion=pos, p=neg_prob)

            users += [u]*pos_len
            pos_items += pos.tolist()
            neg_items += neg.tolist()

        return get_data_iterator(users, pos_items, neg_items, batch_size=self.batch_size, shuffle=True)
