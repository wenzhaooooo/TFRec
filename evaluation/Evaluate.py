'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import heapq  # for retrieval topK
import numpy as np
import math

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

_model = None
_testRatings = None
_testNegatives = None
_K = None
def evaluate(model,evaluateRatings,evaluateNegatives, K,isneupr=False):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _evaluateRatings
    global _evaluateNegatives
    global _K
    _model = model
    _evaluateRatings = evaluateRatings
    _evaluateNegatives = evaluateNegatives
    _K = K

    hits, ndcgs,aucs = [], [],[]
    # Single thread
    for u in range(len(evaluateRatings)):
        if isneupr :
            (hr, ndcg) = eval_neupr(u)
        else :
            (hr, ndcg,auc) = eval_by_user(u)
            aucs.append(auc)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits, ndcgs,aucs)

def eval_neupr(u):
    rating = _evaluateRatings[u]
    items = _evaluateNegatives[u]
    gtItem = rating[1]
    if gtItem==-1:
        return -1,-1
    # items.append(gtItem)
    # Get prediction scores

    users = np.full(len(items), u, dtype='int32')
    item_pos=np.full(len(items),gtItem,dtype='int32')

    predictions1 = _model.sess.run((_model.out), feed_dict={
        _model.user_input: users,
        _model.item_input_pos: item_pos,
        _model.item_input_neg: items,
    })

    predictions2 = _model.sess.run((_model.out), feed_dict={
        _model.user_input: users,
        _model.item_input_pos: items,
        _model.item_input_neg: item_pos,
    })
    prediction=predictions1-predictions2
    num_err=len(prediction[prediction<0])
    if num_err>=_K:
        hr=0
        ndcg=0
    else:
        hr=1
        ndcg=math.log(2) / math.log(num_err + 2)

    return (hr, ndcg)

def eval_by_user(u):
    rating = _evaluateRatings[u]
    items = _evaluateNegatives[u]
    gtItem = rating[1]
    # Get prediction scores
    map_item_score = {}
    predictions = _model.predict(u,gtItem,items)
    for i in np.arange(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    
    auc = []
    neg_predict, pos_predict = predictions[:-1], predictions[-1]
    position = (neg_predict >= pos_predict).sum()
    for _ in range(1, _K + 1):
        # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]
        auc.append(1 - (position / len(neg_predict))) 
    return (hr, ndcg,auc)