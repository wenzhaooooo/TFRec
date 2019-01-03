#ifndef C_TOOLS_H
#define C_TOOLS_H

#include "ThreadPool.h"
#include <vector>
#include <algorithm>
using std::vector;


void c_top_k_index(float *ratings, int rating_len, int top_k, int *result)
{
    vector<int> index(rating_len);
    for(auto i=0; i<rating_len; ++i)
    {
        index[i] = i;
    }
    std::partial_sort_copy(index.begin(), index.end(), result, result+top_k,
                            [& ratings](int &x1, int &x2)->bool{return ratings[x1]>ratings[x2];});
}


void c_top_k_array_index(float **ratings, int rating_len, int rows_num, int top_k, int thread_num, int **results)
{
    ThreadPool pool(thread_num);
    //vector< future< vector<float> > > sync_pre_results;
    for(int i=0; i<rows_num; ++i)
    {
        pool.enqueue(c_top_k_index, ratings[i], rating_len, top_k, results[i]);
    }
}

#endif