#ifndef EVALUATE_H
#define EVALUATE_H

#include "metrics.h"
#include "thread_pool.h"
#include <vector>
#include <future>

using std::vector;
using std::future;


void evaluate(int test_num,
              int **ranks, int rank_len,
              int **ground_truths, int *ground_truths_num,
              int thread_num, float *results)
{
    ThreadPool pool(thread_num);
    vector< future< vector<float> > > sync_pre_results;
    vector< future< vector<float> > > sync_recall_results;
    vector< future< vector<float> > > sync_ap_results;
    vector< future< vector<float> > > sync_ndcg_results;
    vector< future< vector<float> > > sync_mrr_results;
    
    for(int uid=0; uid<test_num; uid++)
    {
        sync_pre_results.emplace_back(pool.enqueue(precision, ranks[uid], rank_len, ground_truths[uid], ground_truths_num[uid]));
        sync_recall_results.emplace_back(pool.enqueue(recall, ranks[uid], rank_len, ground_truths[uid], ground_truths_num[uid]));
        sync_ap_results.emplace_back(pool.enqueue(ap, ranks[uid], rank_len, ground_truths[uid], ground_truths_num[uid]));
        sync_ndcg_results.emplace_back(pool.enqueue(ndcg, ranks[uid], rank_len, ground_truths[uid], ground_truths_num[uid]));
        sync_mrr_results.emplace_back(pool.enqueue(mrr, ranks[uid], rank_len, ground_truths[uid], ground_truths_num[uid]));
    }
    
    float *pre_results = results + 0*rank_len;
    float *recall_results = results + 1*rank_len;
    float *ap_results = results + 2*rank_len;
    float *ndcg_results = results + 3*rank_len;
    float *mrr_results = results + 4*rank_len;

    for(auto && result: sync_pre_results)
    {
        auto tmp_result = result.get();
        for(int k=0; k<rank_len; k++)
        {
            pre_results[k] += tmp_result[k];
        }
    }

    for(auto && result: sync_recall_results)
    {
        auto tmp_result = result.get();
        for(int k=0; k<rank_len; k++)
        {
            recall_results[k] += tmp_result[k];
        }
    }

    for(auto && result: sync_ap_results)
    {
        auto tmp_result = result.get();
        for(int k=0; k<rank_len; k++)
        {
            ap_results[k] += tmp_result[k];
        }
    }

    for(auto && result: sync_ndcg_results)
    {
        auto tmp_result = result.get();
        for(int k=0; k<rank_len; k++)
        {
            ndcg_results[k] += tmp_result[k];
        }
    }

    for(auto && result: sync_mrr_results)
    {
        auto tmp_result = result.get();
        for(int k=0; k<rank_len; k++)
        {
            mrr_results[k] += tmp_result[k];
        }
    }

    for(int k=0; k<rank_len; k++)
    {
        pre_results[k] /= test_num;
        recall_results[k] /= test_num;
        ap_results[k] /= test_num;
        ndcg_results[k] /= test_num;
        mrr_results[k] /= test_num;
    }
}

#endif