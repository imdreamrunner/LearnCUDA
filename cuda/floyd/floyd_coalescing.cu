// includes CUDA
#include <cuda_runtime.h>

#include "floyd.h"


#define MAX_BLOCK_SIZE 512


__device__ inline void update_distance(const int size_mat, const int i, 
                        const int j, const int k, int *mat_global) {
    int i0 = i * size_mat + j;
    int i1 = i * size_mat + k;
    int i2 = k * size_mat + j;
    if (mat_global[i1] != -1 && mat_global[i2] != -1) {
        int sum = (mat_global[i1] + mat_global[i2]);
        if (mat_global[i0] == -1 || sum < mat_global[i0]) mat_global[i0] = sum;
    }
}


__global__ void update_mat_on_k(const int size_mat, const int k, int *mat_global) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = index / size_mat;
    const int j = index % size_mat;
    update_distance(size_mat, i, j, k, mat_global);
}


void PL_APSP(int *mat, const size_t size_mat) {
    int *mat_global;

    int num_node = size_mat * size_mat;
    int block_size = min(size_mat, (size_t) MAX_BLOCK_SIZE);
    int num_block = num_node / block_size;

    cudaMalloc(&mat_global, sizeof(int) * num_node);
    cudaMemcpy(mat_global, mat, sizeof(int) * num_node, cudaMemcpyHostToDevice);

    dim3 dimGrid(num_block, 1, 1);
    dim3 dimBlock(block_size, 1, 1);

    for (int k = 0; k < size_mat; k++) {
        update_mat_on_k<<<dimGrid, dimBlock>>>(size_mat, k, mat_global);
    }

    cudaMemcpy(mat, mat_global, sizeof(int) * num_node, cudaMemcpyDeviceToHost);
}
