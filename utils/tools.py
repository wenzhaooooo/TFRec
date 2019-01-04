import numpy as np
import time


def timer(func):
    """The timer decorator
    """
    def inner(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("%s function cost: %fs" % (func.__name__, end_time - start_time))
        return result
    return inner


def random_choice(a, size=None, replace=True, p=None, exclusion=None):
    if exclusion is not None:
        if p is None:
            p = np.ones_like(a)
        else:
            p = np.array(p, copy=True)
        p = np.ndarray.flatten(p)
        p[exclusion] = 0
        p = p / np.sum(p)
    sample = np.random.choice(a, size=size, replace=replace, p=p)
    return sample


def csr_to_dict(sparse_matrix_data):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    idx_value_dict = {}
    for idx, value in enumerate(sparse_matrix_data):
        if any(value.indices):
            idx_value_dict[idx] = value.indices
    return idx_value_dict
