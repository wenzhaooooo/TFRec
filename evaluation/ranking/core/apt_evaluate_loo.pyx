# distutils: language = c++
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
import os

cdef extern from "c_tools.h":
    void c_top_k_array_index(float **ratings, int rating_len, int rows_num,
                             int top_k, int thread_num, int **results)

cdef extern from "evaluate_loo.h":
    void evaluate_loo(int test_num,
                      int **ranks, int rank_len,
                      int *ground_truths,
                      int thread_num, float *results)

def apt_evaluate_loo(ratings, test_items, rank_len = 50, thread_num = None):
    metrics_num = 2

    tests_num, rating_len = np.shape(ratings)
    if tests_num != len(test_items):
        raise Exception("The lengths of 'ranks' and 'test_items' are different.")
    thread_num = (thread_num or (os.cpu_count() or 1) * 5)

    #get ratings pointer
    cdef float **ratings_pt = <float **> PyMem_Malloc(tests_num * sizeof(float *))
    for i in range(tests_num):
        ratings_pt[i] = <float *>np.PyArray_DATA(ratings[i])

    #store ranks results
    ranks = np.zeros([tests_num, rank_len], dtype=np.intc)
    cdef int **ranks_pt = <int **> PyMem_Malloc(tests_num * sizeof(int *))
    for i in range(len(ranks)):
        ranks_pt[i] = <int *>np.PyArray_DATA(ranks[i])

    #get top k rating index
    c_top_k_array_index(ratings_pt, rating_len, tests_num, rank_len, thread_num, ranks_pt)

    # ground truth pointer, the length array of ground truth pointer
    if (not isinstance(test_items, np.ndarray)) and (test_items.dtype != np.intc):
        tmp_test_items = np.array(test_items, dtype=np.intc)
    else:
        tmp_test_items = test_items
    test_items_pt = <int *>np.PyArray_DATA(tmp_test_items)

    #evaluate results

    results = np.zeros([metrics_num*rank_len], dtype=np.float32)
    results_pt = <float *>np.PyArray_DATA(results)

    #evaluate
    evaluate_loo(tests_num, ranks_pt, rank_len, test_items_pt, thread_num, results_pt)

    #release the allocated space
    PyMem_Free(ratings_pt)
    PyMem_Free(ranks_pt)

    metrics_value = np.reshape(results, newshape=[metrics_num, -1])
    return metrics_value
