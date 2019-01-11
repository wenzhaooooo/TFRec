import numpy as np


def ndcg(rank, ground_truth):
    dcg = 0.0
    for idx, item in enumerate(rank):
        if item == ground_truth:
            dcg = 1.0/np.log2(idx+2)
            break
    result = np.zeros(len(rank), dtype=np.float32)
    result[idx:] = dcg
    return result


def hr(rank, ground_truth):
    hit = 0.0
    for idx, item in enumerate(rank):
        if item == ground_truth:
            hit = 1.0
            break
    result = np.zeros(len(rank), dtype=np.float32)
    result[idx:] = hit
    return result
