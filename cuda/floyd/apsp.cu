// includes, system
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
// #include <helper_cuda.h>
// #include <helper_functions.h> // helper functions for SDK examples

#include "MatUtil.h"
#include "floyd.h"

using namespace std;

unsigned long long time_diff(const struct timeval& tv1, const struct timeval& tv2) {
    return (tv2.tv_sec-tv1.tv_sec)*1000000 + tv2.tv_usec-tv1.tv_usec;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        return -1;
    }

    size_t size_mat = atoi(argv[1]);
    size_t num_node = size_mat * size_mat;

    cout << "N = " << size_mat << endl;

    int *mat = (int*)malloc(sizeof(int) * num_node);
    int *ans = (int*)malloc(sizeof(int) * num_node);

    GenMatrix(mat, size_mat);
    memcpy(ans, mat, sizeof(int) * num_node);

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    ST_APSP(ans, size_mat);
    gettimeofday(&end_time, NULL);
    int sequential_time = time_diff(start_time, end_time);

    gettimeofday(&start_time, NULL);
    PL_APSP(mat, size_mat);
    gettimeofday(&end_time, NULL);
    int parallel_time = time_diff(start_time, end_time);

    cout << "Sequential time: " << sequential_time << " ns" << endl;
    cout << "  Parallel time: " << parallel_time << " ns" << endl;
    cout << "        Speedup: " << (1.0 * sequential_time / parallel_time) << endl;

    if (CmpArray(mat, ans, num_node)) {
        cout << "Parallel answer is correct!" << endl;
    } else {
        cout << "Parallel answer is wrong!" << endl;
    }
}
