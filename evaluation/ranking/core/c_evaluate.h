#ifndef C_EVALUATE_H
#define C_EVALUATE_H

#include "c_metrics.h"
#include "ThreadPool.h"
#include <vector>
#include <future>
using std::vector;
using std::future;

//declaration
void c_evaluate(int test_num, int **ranks, int rank_len, int **ground_truths, int *ground_truths_num, int thread_num, float *results);

//implementation


void c_evaluate(int test_num,
                int **ranks, int rank_len,
                int **ground_truths, int *ground_truths_num,
                int thread_num, float *results)
{
    ThreadPool pool(thread_num);
    vector< future< vector<float> > > sync_pre_results;
    vector< future< vector<float> > > sync_recall_results;
    vector< future< vector<float> > > sync_ap_results;
    vector< future< vector<float> > > sync_ndcg_top_results;
    vector< future< vector<float> > > sync_ndcg_all_results;
    vector< future< vector<float> > > sync_mrr_results;

    for(int uid=0; uid<test_num; uid++)
    {
        sync_pre_results.emplace_back(pool.enqueue(c_precision, ranks[uid], rank_len, ground_truths[uid], ground_truths_num[uid]));
        sync_recall_results.emplace_back(pool.enqueue(c_recall, ranks[uid], rank_len, ground_truths[uid], ground_truths_num[uid]));
        sync_ap_results.emplace_back(pool.enqueue(c_ap, ranks[uid], rank_len, ground_truths[uid], ground_truths_num[uid]));
        sync_ndcg_top_results.emplace_back(pool.enqueue(c_ndcg_top, ranks[uid], rank_len, ground_truths[uid], ground_truths_num[uid]));
        sync_ndcg_all_results.emplace_back(pool.enqueue(c_ndcg_all, ranks[uid], rank_len, ground_truths[uid], ground_truths_num[uid]));
        sync_mrr_results.emplace_back(pool.enqueue(c_mrr, ranks[uid], rank_len, ground_truths[uid], ground_truths_num[uid]));
    }
    
    float *pre_results = results + 0*rank_len;
    float *recall_results = results + 1*rank_len;
    float *ap_results = results + 2*rank_len;
    float *ndcg_top_results = results + 3*rank_len;
    float *ndcg_all_results = results + 4*rank_len;
    float *mrr_results = results + 5*rank_len;

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

    for(auto && result: sync_ndcg_top_results)
    {
        auto tmp_result = result.get();
        for(int k=0; k<rank_len; k++)
        {
            ndcg_top_results[k] += tmp_result[k];
        }
    }

    for(auto && result: sync_ndcg_all_results)
    {
        auto tmp_result = result.get();
        for(int k=0; k<rank_len; k++)
        {
            ndcg_all_results[k] += tmp_result[k];
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
        ndcg_top_results[k] /= test_num;
        ndcg_all_results[k] /= test_num;
        mrr_results[k] /= test_num;
    }
}

#endif