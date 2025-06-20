#include <cuda_runtime.h>

// A simple CUDA kernel to add two vectors element-wise.
__global__ void add_vectors_kernel(float *out, const float *a, const float *b,
                                   int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

// C++ wrapper for launching the CUDA kernel.
void launch_add_vectors(float *out, const float *a, const float *b, int n) {
    if (n <= 0)
        return;
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    add_vectors_kernel<<<blocks_per_grid, threads_per_block>>>(out, a, b, n);
}
