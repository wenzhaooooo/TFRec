#ifndef C_METRICS_H
#define C_METRICS_H

#include <vector>
#include <set>
#include <cmath>
using std::vector;
using std::set;


vector<float> c_precision(int *rank, int top_k, int *truth, int truth_len)
{
    vector<float> result(top_k);
    int hits = 0;
    set<int> truth_set(truth, truth+truth_len);
    for(int i=0; i<top_k; i++)
    {
        if(truth_set.count(rank[i]))
        {
            hits += 1;
        }
        result[i] = 1.0*hits / (i+1);
    }
    return result;
}

vector<float> c_recall(int *rank, int top_k, int *truth, int truth_len)
{
    vector<float> result(top_k);
    int hits = 0;
    set<int> truth_set(truth, truth+truth_len);
    for(int i=0; i<top_k; i++)
    {
        if(truth_set.count(rank[i]))
        {
            hits += 1;
        }
        result[i] = 1.0*hits / truth_len;
    }
    return result;
}

vector<float> c_ap(int *rank, int top_k, int *truth, int truth_len)
{
    vector<float> result = c_precision(rank, top_k, truth, truth_len);
    set<int> truth_set(truth, truth+truth_len);
    float sum_pre = 0;
    for(int i=0; i<top_k; i++)
    {
        if(truth_set.count(rank[i]))
        {
            sum_pre += result[i];
        }
        result[i] = 1.0*sum_pre/(std::min(truth_len, i+1));
    }
    return result;
}

vector<float> c_ndcg_top(int *rank, int top_k, int *truth, int truth_len)
{
    vector<float> result(top_k);
    float iDCG = 0;
    float DCG = 0;
    set<int> truth_set(truth, truth+truth_len);
    for(int i=0; i<top_k; i++)
    {
        if(truth_set.count(rank[i]))
        {
            DCG += 1.0/log2(i+2);
        }
        iDCG += 1.0/log2(i+2);
        result[i] = DCG/iDCG;
    }
    return result;
}

vector<float> c_ndcg_all(int *rank, int top_k, int *truth, int truth_len)
{
    vector<float> result(top_k);
    float iDCG = 0;
    for(int i=2; i<truth_len+2; i++)
    {
        iDCG += 1.0/log2(i);
    }
    set<int> truth_set(truth, truth+truth_len);
    float DCG = 0;
    for(int i=0; i<top_k; i++)
    {
        if(truth_set.count(rank[i]))
        {
            DCG += 1.0/log2(i+2);
        }
        result[i] = DCG/iDCG;
    }
    return result;
}

vector<float> c_mrr(int *rank, int top_k, int *truth, int truth_len)
{
    vector<float> result(top_k);
    float rr = 0;
    set<int> truth_set(truth, truth+truth_len);
    for(int i=0; i<top_k; i++)
    {
        if(truth_set.count(rank[i]))
        {
            rr = 1.0/(i+1);
            for(int j=i; j<top_k; j++)
            {
                result[j] = rr;
            }
            break;
        }
        else
        {
            rr = 0.0;
            result[i] =rr;
        }
    }
    return result;
}

#endif