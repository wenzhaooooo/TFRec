#ifndef EVALUATE_LOO_H
#define EVALUATE_LOO_H

#include "loo_metrics.h"
#include "ThreadPool.h"
#include <vector>
#include <future>
using std::vector;
using std::future;

//implementation
void evaluate_loo(int test_num,
                  int **ranks, int rank_len,
                  int *ground_truths,
                  int thread_num, float *results)
{
    ThreadPool pool(thread_num);
    vector< future< vector<float> > > sync_ndcg_results;
    vector< future< vector<float> > > sync_hr_results;

    for(int uid=0; uid<test_num; uid++)
    {
        sync_ndcg_results.emplace_back(pool.enqueue(ndcg, ranks[uid], rank_len, ground_truths[uid]));
        sync_hr_results.emplace_back(pool.enqueue(hit_ratio, ranks[uid], rank_len, ground_truths[uid]));
    }
    
    float *ndcg_results = results + 0*rank_len;
    float *hr_results = results + 1*rank_len;

    for(auto && result: sync_ndcg_results)
    {
        auto tmp_result = result.get();
        for(int k=0; k<rank_len; k++)
        {
            ndcg_results[k] += tmp_result[k];
        }
    }

    for(auto && result: sync_hr_results)
    {
        auto tmp_result = result.get();
        for(int k=0; k<rank_len; k++)
        {
            hr_results[k] += tmp_result[k];
        }
    }

    for(int k=0; k<rank_len; k++)
    {
        ndcg_results[k] /= test_num;
        hr_results[k] /= test_num;
    }
}

#endif