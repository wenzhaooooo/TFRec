import numpy as np


def precision(rank, ground_truth):
    # TODO ensure the denominator
    hits = [1 if item in ground_truth else 0 for item in rank]
    result = np.cumsum(hits, dtype=np.float)/np.arange(1, len(rank)+1)
    return result


def recall(rank, ground_truth):
    hits = [1 if item in ground_truth else 0 for item in rank]
    result = np.cumsum(hits, dtype=np.float) / len(ground_truth)
    return result


def map(rank, ground_truth):
    # TODO ensure the denominator
    pre = precision(rank, ground_truth)
    pre = [pre[idx] if item in ground_truth else 0 for idx, item in enumerate(rank)]
    sum_pre = np.cumsum(pre, dtype=np.float32)
    gt_len = len(ground_truth)
    len_rank = np.array([min(i, gt_len) for i in range(1, len(rank)+1)])
    result = sum_pre/len_rank
    return result


def ndcg(rank, ground_truth):
    len_rank = len(rank)
    idcg = np.cumsum(1.0/np.log2(np.arange(2, len_rank+2)))
    dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
    result = dcg/idcg
    return result


def mrr(rank, ground_truth):
    for idx, item in enumerate(rank):
        if item in ground_truth:
            break
    result = np.zeros(len(rank), dtype=np.float32)
    result[idx:] = 1.0/(idx+1)
    return result
