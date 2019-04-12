import numpy as np
from utils import typeassert
from .utils import load_data, filter_data, remap_id
from .utils import split_by_ratio, split_by_loo


class Splitter(object):
    @typeassert(config=dict)
    def __init__(self, config):
        self.filename = config["data_file"]
        self.ratio = eval(config["ratio"])
        self.file_format = config["file_format"]
        self.sep = eval(config["separator"])
        self.user_min = eval(config["user_min"])
        self.item_min = eval(config["item_min"])
        self.by_time = eval(config["by_time"])
        self.spliter = config["splitter"]

    def split(self):
        if self.file_format.lower() == "uirt":
            columns = ["user", "item", "rating", "time"]
            if self.by_time is False:
                by_time = False
            else:
                by_time = True
        elif self.file_format.lower() == "uir":
            columns = ["user", "item", "rating"]
            by_time = False
        else:
            raise ValueError("There is not data format '%s'" % self.file_format)

        all_data = load_data(self.filename, sep=self.sep, columns=columns)
        filtered_data = filter_data(all_data, user_min=self.user_min, item_min=self.item_min)
        remapped_data, user2id, item2id = remap_id(filtered_data)

        if self.spliter == "ratio":
            train_data, test_data = split_by_ratio(remapped_data, ratio=self.ratio, by_time=by_time)
        elif self.spliter == "loo":
            train_data, test_data = split_by_loo(remapped_data, by_time=by_time)
        else:
            raise ValueError("There is not splitter '%s'" % self.spliter)

        np.savetxt(self.filename+".train", train_data, fmt="%d", delimiter="\t")
        np.savetxt(self.filename+".test", test_data, fmt="%d", delimiter="\t")

        user_id = [[str(user), str(id)]for user, id in user2id.items()]
        np.savetxt(self.filename+".user2id", user_id, fmt="%s", delimiter="\t")

        item_id = [[str(item), str(id)] for item, id in item2id.items()]
        np.savetxt(self.filename + ".item2id", item_id, fmt="%s", delimiter="\t")
