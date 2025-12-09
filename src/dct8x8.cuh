// dct8x8_kernels.cuh
// Public interface for 8x8 block-wise DCT (CPU reference implementations + CUDA kernels)

#pragma once

#include <cuda_runtime.h>

// ------------------------- GPU constants initialization -------------------------

// Initialize cosine lookup table for N=8 in GPU constant memory.
// The table stores cos((2n+1)kπ/16) for k,n in [0..7].
void init_cos_table_8_gpu();

// Initialize the 7 constants required by Lee’s algorithm for N=8,
// stored in GPU constant memory.
void init_lee_table_8_gpu();

// ------------------------- CPU reference implementations -------------------------

// Naive 8x8 DCT (double-sum) applied to each 8x8 block.
// Computes the full 2D DCT coefficients directly.
// Used as ground truth for validation.
void dct8x8_cpu_naive_reference(const float* input,
                                   float* output,
                                   int width, int height);

// Separable 8x8 DCT (row-wise 1D DCT followed by column-wise 1D DCT).
// Complexity is O(N^3) per image, faster than the naive double-sum.
// Closer to how GPU kernels 2/3 are structured.
void dct8x8_cpu_separable(const float* input,
                                   float* output,
                                   int width, int height);

// 8x8 DCT using Lee’s factorization algorithm.
// Performs two passes of 1D Lee DCT with an intermediate transpose.
// Computationally efficient and used as a reference for kernel 4.
void dct8x8_cpu_lee(const float* input,
                             float* output,
                             int width, int height);

// ------------------------- Utility -------------------------

// Compare two arrays of equal size and print detailed error statistics.
// Returns true if all elements satisfy abs(a-b) <= atol + rtol * |b|.
bool compare_arrays_and_print(const char* tag,
                              const float* a,
                              const float* b,
                              size_t numel,
                              int width,
                              float atol,
                              float rtol,
                              int max_print = 10);

// ------------------------- CUDA kernels -------------------------

// Kernel 0: Naive direct 2D DCT (double-sum), using only global memory.
// Very high arithmetic cost and many redundant memory accesses.
__global__ void dct8x8_gpu_naive(const float* __restrict__ input,
                              float* __restrict__ output,
                              int width, int height);

// Kernel 1: Same as Kernel 0, but first loads an 8x8 tile into shared memory
// to increase arithmetic intensity and reduce global memory traffic.
__global__ void dct8x8_gpu_naive_shared_memory(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            int width, int height);

// Kernel 2: Separable DCT (row, then column) using global scratch buffer
// for intermediate results. Simplifies implementation but expensive
// due to global memory writes and reads.
__global__ void dct8x8_gpu_naive_separable(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        float* __restrict__ scratch, // same size as input image
                                        int width, int height);

// Kernel 3: Separable DCT using shared memory + shared transpose.
// Avoids global intermediate storage and improves locality.
__global__ void dct8x8_gpu_naive_separable_shared_memory(const float* __restrict__ input,
                                                      float* __restrict__ output,
                                                      int width, int height);

// Kernel 4: Separable DCT using shared memory + shared transpose,
// combined with Lee’s 1D DCT factorization algorithm for higher efficiency.
__global__ void dct8x8_gpu_lee8_shared(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             int width, int height);
