import numpy as np
from utils.tools import random_choice
from data.DataIterator import get_data_iterator
from model.BPR import BPR


class DNSBPR(BPR):
    def __init__(self, sess, dataset):
        super(DNSBPR, self).__init__(sess, dataset)
        self.sample_num = int(self.conf["sample_num"])

    def get_training_data(self):
        users = []
        pos_items = []
        neg_items = []
        for u, pos in self.user_pos_train.items():
            pos_len = len(pos)
            feed = {self.user_h: [u]}
            logits = self.sess.run(self.all_logits, feed_dict=feed)
            logits = np.reshape(logits, newshape=[-1])

            neg_pool = random_choice(self.all_items, size=self.sample_num*pos_len, exclusion=pos)

            neg_logits = logits[neg_pool]

            neg_pool = np.reshape(neg_pool, newshape=[pos_len, self.sample_num])
            neg_logits = np.reshape(neg_logits, newshape=[pos_len, self.sample_num])

            neg = neg_pool[np.arange(pos_len), np.argmax(neg_logits, axis=1)]

            users += [u]*pos_len
            pos_items += pos.tolist()
            neg_items += neg.tolist()

        return get_data_iterator(users, pos_items, neg_items, batch_size=self.batch_size, shuffle=True)
