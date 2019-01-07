# distutils: language = c++
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
import os

cdef extern from "c_tools.h":
    void c_top_k_array_index(float **ratings, int rating_len, int rows_num,
                             int top_k, int thread_num, int **results)

# TODO rename function and file
cdef extern from "c_evaluate.h":
    void c_evaluate(int test_num,
                    int **ranks, int rank_len,
                    int **ground_truths, int *ground_truths_num,
                    int thread_num, float *results)

def apt_evaluate(ratings, test_items, top_k=50, thread_num = None):
    metrics_num = 6
    tests_num, rating_len = np.shape(ratings)
    if tests_num != len(test_items):
        raise Exception("The lengths of 'ranks' and 'test_items' are different.")
    thread_num = (thread_num or (os.cpu_count() or 1) * 5)

    #get ratings pointer
    cdef float **ratings_pt = <float **> PyMem_Malloc(tests_num * sizeof(float *))
    for i in range(tests_num):
        ratings_pt[i] = <float *>np.PyArray_DATA(ratings[i])

    #store ranks results
    ranks = np.zeros([tests_num, top_k], dtype=np.intc)
    cdef int **ranks_pt = <int **> PyMem_Malloc(tests_num * sizeof(int *))
    for i in range(len(ranks)):
        ranks_pt[i] = <int *>np.PyArray_DATA(ranks[i])

    #get top k rating index
    c_top_k_array_index(ratings_pt, rating_len, tests_num, top_k, thread_num, ranks_pt)

    # ground truth pointer, the length array of ground truth pointer
    test_items_pt = <int **> PyMem_Malloc(tests_num * sizeof(int *))
    test_num_pt = <int *> PyMem_Malloc(tests_num * sizeof(int))
    for u in range(tests_num):
        test_items_pt[u] = <int *>np.PyArray_DATA(test_items[u])
        test_num_pt[u] = len(test_items[u])

    #evaluate results
    results = np.zeros([metrics_num * top_k], dtype=np.float32)
    results_pt = <float *>np.PyArray_DATA(results)

    #evaluate
    c_evaluate(tests_num, ranks_pt, top_k, test_items_pt, test_num_pt, thread_num, results_pt)

    #release the allocated space
    PyMem_Free(ratings_pt)
    PyMem_Free(ranks_pt)
    PyMem_Free(test_items_pt)
    PyMem_Free(test_num_pt)

    metrics_value = np.reshape(results, newshape=[metrics_num, -1])
    return metrics_value
