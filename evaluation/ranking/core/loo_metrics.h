#ifndef LOO_METRICS_H
#define LOO_METRICS_H

#include <vector>
#include <set>
#include <cmath>
using std::vector;


vector<float> ndcg(int *rank, int top_k, int ground_truth)
{
    vector<float> result(top_k);
    float DCG = 0.0;
    for(int i=0; i<top_k; i++)
    {
        if(rank[i]==ground_truth)
        {
            DCG = 1.0/log2(i+2);
            for(int j=i; j<top_k; j++)
            {
                result[j] = DCG;
            }
            break;
        }
        else
        {
            result[i] = 0.0;
        }
    }
    return result;
}

vector<float> hit_ratio(int *rank, int top_k, int ground_truth)
{
    vector<float> result(top_k);
    for(int i=0; i<top_k; i++)
    {
        if(rank[i]==ground_truth)
        {
            for(int j=i; j<top_k; j++)
            {
                result[j] = 1.0;
            }
            break;
        }
        else
        {
            result[i] =0.0;
        }
    }
    return result;
}

#endif