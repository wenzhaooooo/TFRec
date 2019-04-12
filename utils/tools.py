import numpy as np
import time
import heapq
import itertools
from inspect import signature
from functools import wraps
from scipy.sparse import csr_matrix


def typeassert(*type_args, **type_kwargs):
    def decorate(func):
        sig = signature(func)
        bound_types = sig.bind_partial(*type_args, **type_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError('Argument {} must be {}'.format(name, bound_types[name]))
            return func(*args, **kwargs)
        return wrapper
    return decorate


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


@typeassert(sparse_matrix_data=csr_matrix)
def csr_to_user_dict(sparse_matrix_data):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    idx_value_dict = {}
    for idx, value in enumerate(sparse_matrix_data):
        if any(value.indices):
            idx_value_dict[idx] = value.indices
    return idx_value_dict


@typeassert(sparse_matrix_data=csr_matrix)
def csr_to_user_item_pair(sparse_matrix_data):
    users, items = [], []
    for user, u_items in enumerate(sparse_matrix_data):
        items_num = u_items.nnz
        users.extend([user]*items_num)
        items.extend(u_items.indices)
    return users, items


def argmax_top_k(a, top_k=50):
    ele_idx = heapq.nlargest(top_k, zip(a, itertools.count()))
    return np.array([idx for ele, idx in ele_idx], dtype=np.intc)
