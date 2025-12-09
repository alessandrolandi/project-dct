//  dct8x8_kernels.cu
//  All DCT function/kernel do 8x8 DCT with no scaling (unscaled), since JPEG compression absorb scaling into quantization step.
//  Test on 1024x1024 image (8x8 blocks dct opertaion).
//  compare results with dct8x8_cpu_naive_reference to verify correctness.
//  brun ./dct_8x8 arg1, arg1=0..4 to select GPU kernel to run.
//
//  CPU:
//  Three CPU reference implementations available:
//  dct8x8_cpu_naive_reference    : naive direct 2D DCT (double-sum) CPU reference
//  dct8x8_cpu_separable   : separable DCT (row 1D then col 1D) CPU reference
//  dct8x8_cpu_lee        : Lee's factorization DCT CPU reference
//  GPU:
//  Five kernels available:
//  0) dct8x8_gpu_naive                         : direct 2D DCT double-sum (global mem)
//  1) dct8x8_gpu_naive_shared_memory           : direct 2D double-sum but load tile into shared first
//  2) dct8x8_gpu_naive_separable               : separable (row 1D then col 1D) using GLOBAL scratch (no shared memory used)
//  3) dct8x8_gpu_naive_separable_shared_memory : separable using shared + shared transpose
//  4) dct8x8_gpu_lee8_shared          : Lee's factorization using shared + shared transpose
//  Adds CPU reference (unscaled) and validates GPU output vs CPU with epsilon.

#include "dct8x8.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <limits>
#include <algorithm>
#include <chrono>
#include <utility>  // std::swap
#include <cstring>  // std::memcpy

// Macro Switches
// = 1: we compare GPU Lee8 kernel output with CPU Lee8 output
#ifndef RUN_CPU_LEE8
#define RUN_CPU_LEE8 1
#endif

#define PI 3.14159265358979323846f

#define CHECK_CUDA(call) do {                                                    \
    cudaError_t err = (call);                                                    \
    if (err != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                                \
                __FILE__, __LINE__, cudaGetErrorString(err));                    \
        std::exit(EXIT_FAILURE);                                                 \
    }                                                                            \
} while (0)

// ---------------- CPU stuff initialization ----------------
void init_cos_table_8_cpu(float cos8[64])
{
    for (int k = 0; k < 8; ++k) {
        for (int n = 0; n < 8; ++n) {
            cos8[k * 8 + n] = std::cos(PI * (2.0f * n + 1.0f) * k / 16.0f);
        }
    }
}

// easy access function
inline float COS8_CPU(const float cos8[64], int k, int n)
{
    return cos8[k * 8 + n];
}

// Lee DCT needs 7 constants for 1ddct on N=8:
// c[0..3] = 0.5 / cos((2*i+1)*pi/16), i=0..3
// c[4..5] = 0.5 / cos((2*i+1)*pi/8),  i=0..1
// c[6]    = 0.5 / cos(pi/4)
// 7 floats total, 4*7=28 bytes
void init_lee_table_8_cpu(float lee8[7])
{
    // use double precision for better accuracy
    const double PI_d = 3.1415926535897932384626433832795;
    const double phases[7] = {
        PI_d / 16.0,
        3.0 * PI_d / 16.0,
        5.0 * PI_d / 16.0,
        7.0 * PI_d / 16.0,
        PI_d / 8.0,
        3.0 * PI_d / 8.0,
        PI_d / 4.0
    };
    for (int i = 0; i < 7; ++i) {
        lee8[i] = (float)(0.5 / std::cos(phases[i]));
    }
}


// -------------------------------- GPU stuff initialization --------------------------------
// In this case 64*4 bytes = 256 bytes (very small compared to 64 KB constant memory)
// , which I believe fits into L1 constant cache(2 KiB in V100),
// (V100 has 2KiB L1 constant cache), see page 18 and 19 Figure 3.1 and Table 3.1 in https://arxiv.org/pdf/1804.06826
__constant__ float c_cos8[64];

// Precomputed cos table for N=8:
// cos((2*n+1)*k*pi/16) where k in [0..7], n in [0..7]
// constant memory should be alocated and transferred from host
void init_cos_table_8_gpu()
{
    float h[64];
    for (int k = 0; k < 8; ++k) {
        for (int n = 0; n < 8; ++n) {
            h[k * 8 + n] = std::cos(PI * (2.0f * n + 1.0f) * k / 16.0f);
        }
    }
    CHECK_CUDA(cudaMemcpyToSymbol(c_cos8, h, sizeof(h)));
}

// Layout: c_cos8[k*8 + n]
__device__ __forceinline__ float COS8(int k, int n)
{
    return c_cos8[k * 8 + n];
}

// Lee DCT needs 7 constants for 1ddct on N=8:
// c[0..3] = 0.5 / cos((2*i+1)*pi/16), i=0..3
// c[4..5] = 0.5 / cos((2*i+1)*pi/8),  i=0..1
// c[6]    = 0.5 / cos(pi/4)
// 7 floats total, 4*7=28 bytes
__constant__ float c_lee8[7];

void init_lee_table_8_gpu()
{
    float h[7];
    const float phases[7] = {
        PI / 16.0f,
        3.0f * PI / 16.0f,
        5.0f * PI / 16.0f,
        7.0f * PI / 16.0f,
        PI / 8.0f,
        3.0f * PI / 8.0f,
        PI / 4.0f
    };
    for (int i = 0; i < 7; ++i) h[i] = 0.5f / std::cos(phases[i]);
    CHECK_CUDA(cudaMemcpyToSymbol(c_lee8, h, sizeof(h)));
}

// easy access function
__device__ __forceinline__ float LEE8(int i) { return c_lee8[i]; }

// Compare two float arrays (same size) and report mismatches.
// A mismatch is counted when abs(a-b) > atol + rtol * |b|.   Here I drew inspiration from numpy.isclose().
// Prints up to `max_print` detailed mismatch entries and a summary.
// Returns true if all elements are within tolerance.
bool compare_arrays_and_print(const char* tag,
                                     const float* a,
                                     const float* b,
                                     size_t numel,
                                     int width,
                                     float atol,
                                     float rtol,
                                     int max_print)
{
    size_t bad = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t max_idx = 0;

    for (size_t i = 0; i < numel; ++i) {
        float av = a[i];
        float bv = b[i];
        float abs_err = std::fabs(av - bv);
        float thr = atol + rtol * std::fabs(bv);

        // rel_err is only used in print, also max(1,|b|) avoid divided by 0
        float rel_err = abs_err / std::max(1.0f, std::fabs(bv));

        if (abs_err > max_abs) { max_abs = abs_err; max_idx = i; }
        if (rel_err > max_rel) { max_rel = rel_err; }

        if (abs_err > thr) {
            if (bad < (size_t)max_print) {
                int y = (int)(i / (size_t)width);
                int x = (int)(i % (size_t)width);
                printf("[%s] Mismatch[%zu] at (y=%d,x=%d): a=% .6f b=% .6f abs=%g thr=%g rel=%g\n",
                       tag, i, y, x, av, bv, abs_err, thr, rel_err);
            }
            ++bad;
        }
    }

    int y = (int)(max_idx / (size_t)width);
    int x = (int)(max_idx % (size_t)width);
    printf("[%s] bad=%zu / %zu (%.6f%%), max_abs=%g at (y=%d,x=%d), max_rel=%g, atol=%g, rtol=%g\n",
           tag, bad, numel, 100.0 * (double)bad / (double)numel, max_abs, y, x, max_rel, atol, rtol);

    return (bad == 0);
}

// ------------------------------------------------------------
// CPU reference: block-wise 8x8 DCT (unscaled) on 8n x 8n image
// complexity: O(N^2) per coefficient, total O(N^4) per 8x8 block
// For each 8x8 tile, compute all 64 coefficients by direct double sum.
// Use this result as ground truth to validate GPU outputs.
// ------------------------------------------------------------
void dct8x8_cpu_naive_reference(const float* input,
                                         float* output,
                                         int width, int height)
{
    float cos8[64];
    init_cos_table_8_cpu(cos8);

    constexpr int TILE = 8;
    // for each 8x8 block(tile)
    for (int by = 0; by < height; by += TILE) {
        for (int bx = 0; bx < width;  bx += TILE) {

            // compute coefficients (u,v)
            for (int u = 0; u < TILE; ++u) {
                for (int v = 0; v < TILE; ++v) {
                    float sum = 0.0f;
                    for (int y = 0; y < TILE; ++y) {
                        const float cu = COS8_CPU(cos8, u, y);
                        for (int x = 0; x < TILE; ++x) {
                            const float cv = COS8_CPU(cos8, v, x);
                            const float pix = input[(by + y) * width + (bx + x)]; // global read
                            sum += pix * cu * cv;
                        }
                    }
                    output[(by + u) * width + (bx + v)] = sum;
                }
            }
        }
    }
}

// ------------------------------------------------------------
// 1D DCT (unscaled) on length-8 vector, used in separable DCT
// spatial domain(pixel) to frequency domain
// ------------------------------------------------------------
inline void dct1d_cpu_on8(const float in[8],
                                        float out[8],
                                        const float cos8[64])
{
    // k = frequency index 0..7
    for (int k = 0; k < 8; ++k) {
        float sum = 0.0f;
        // n = spatial index 0..7
        for (int n = 0; n < 8; ++n) {
            sum += in[n] * COS8_CPU(cos8, k, n);
        }
        out[k] = sum;
    }
}

// ------------------------------------------------------------
// CPU separable DCT: block-wise 8x8 (unscaled) on 8n x 8n image
// complexity: O(N^2) per 1D DCT, total O(2N\times N^2)=O(N^3), per 8x8 block
// Step 1: apply 1D DCT to each row
// Step 2: apply 1D DCT to each column
// Output same size as input.
// ------------------------------------------------------------
void dct8x8_cpu_separable(const float* input,
                                          float* output,
                                          int width, int height)
{
    float cos8[64];
    init_cos_table_8_cpu(cos8);

    constexpr int TILE = 8;

    for (int by = 0; by < height; by += TILE) {
        for (int bx = 0; bx < width;  bx += TILE) {

            float tile[8][8];
            float tmp[8][8];

            // --- load 8x8 tile from input ---
            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 8; ++x) {
                    tile[y][x] = input[(by + y) * width + (bx + x)];
                }
            }

            // --- Step 1: row-wise 1D DCT ---
            // tmp[y][u] = DCT( tile[y][x] over x ) at frequency u
            for (int y = 0; y < 8; ++y) {
                float in_row[8];
                float out_row[8];
                for (int x = 0; x < 8; ++x) {
                    in_row[x] = tile[y][x];
                }
                dct1d_cpu_on8(in_row, out_row, cos8);
                for (int u = 0; u < 8; ++u) {
                    tmp[y][u] = out_row[u];
                }
            }

            // --- Step 2: column-wise 1D DCT ---
            // For each column v, do 1D DCT along y:
            //   col_in[y] = tmp[y][v], col_out[u] = DCT(col_in)
            // final coefficient (u,v) stored to output[(by+u, bx+v)]
            for (int v = 0; v < 8; ++v) {
                float col_in[8];
                float col_out[8];
                for (int y = 0; y < 8; ++y) {
                    col_in[y] = tmp[y][v];
                }
                dct1d_cpu_on8(col_in, col_out, cos8);
                for (int u = 0; u < 8; ++u) {
                    output[(by + u) * width + (bx + v)] = col_out[u];
                }
            }
        }
    }
}

// 1D dct on length-8 vector using Lee's algorithm
// original paper use recursive approach
// here I translate it to iterative approach, since recursion is not well supported on GPU
inline void dct1d_cpu_lee_on8(float x[8], const float lee_table[7])
{
    float buf0[8], buf1[8];
    float *curr = buf0, *next = buf1;

    // copy in
    #pragma unroll
    for (int i = 0; i < 8; ++i) curr[i] = x[i];

    // ---------- Forward: len=8, half=4 (uses c0..c3) ----------
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a = curr[i];
        float b = curr[7 - i];
        next[i]     = a + b;
        next[i + 4] = (a - b) * lee_table[i];
    }
    std::swap(curr, next);

    // ---------- Forward: len=4, half=2 on segments base=0 and base=4 (uses c4..c5) ----------
    #pragma unroll
    for (int base = 0; base <= 4; base += 4) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            float a = curr[base + i];
            float b = curr[base + (3 - i)];
            next[base + i]     = a + b;
            next[base + i + 2] = (a - b) * lee_table[4 + i];
        }
    }
    std::swap(curr, next);

    // ---------- Forward: len=2, half=1 on segments base=0,2,4,6 (uses c6) ----------
    #pragma unroll
    for (int base = 0; base < 8; base += 2) {
        float a = curr[base + 0];
        float b = curr[base + 1];
        next[base + 0] = a + b;
        next[base + 1] = (a - b) * lee_table[6];
    }
    std::swap(curr, next);

    // ---------- Backward: len=4, half=2 ----------
    // segment base=0
    next[0] = curr[0];
    next[1] = curr[2] + curr[3];
    next[2] = curr[1];
    next[3] = curr[3];
    // segment base=4
    next[4] = curr[4];
    next[5] = curr[6] + curr[7];
    next[6] = curr[5];
    next[7] = curr[7];
    std::swap(curr, next);

    // ---------- Backward: len=8, half=4 ----------
    next[0] = curr[0];
    next[1] = curr[4] + curr[5];
    next[2] = curr[1];
    next[3] = curr[5] + curr[6];
    next[4] = curr[2];
    next[5] = curr[6] + curr[7];
    next[6] = curr[3];
    next[7] = curr[7];
    std::swap(curr, next);

    // copy out
    #pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = curr[i];
}

// use Lee's algorithm to do block-wise 8x8 DCT on CPU
void dct8x8_cpu_lee(const float* input,
                                   float* output,
                                   int width, int height)
{
    float lee_table[7];
    init_lee_table_8_cpu(lee_table);

    constexpr int TILE = 8;

    for (int by = 0; by < height; by += TILE) {
        for (int bx = 0; bx < width;  bx += TILE) {

            float tile[8][8];
            float tr[8][8];

            // load
            for (int y = 0; y < 8; ++y)
                for (int x = 0; x < 8; ++x)
                    tile[y][x] = input[(by + y) * width + (bx + x)];

            // 1) row-wise Lee
            for (int y = 0; y < 8; ++y) {
                float row[8];
                #pragma unroll
                for (int x = 0; x < 8; ++x) row[x] = tile[y][x];
                dct1d_cpu_lee_on8(row, lee_table);
                #pragma unroll
                for (int x = 0; x < 8; ++x) tile[y][x] = row[x];
            }

            // 2) transpose to tr[v][u] = tile[u][v]
            for (int u = 0; u < 8; ++u)
                for (int v = 0; v < 8; ++v)
                    tr[v][u] = tile[u][v];

            // 3) "col-wise" Lee by applying row-wise Lee on transposed matrix rows (indexed by v)
            for (int v = 0; v < 8; ++v) {
                float row[8];
                #pragma unroll
                for (int u = 0; u < 8; ++u) row[u] = tr[v][u];
                dct1d_cpu_lee_on8(row, lee_table);
                #pragma unroll
                for (int u = 0; u < 8; ++u) tr[v][u] = row[u];
            }

            // 4) write back: output[u][v] = tr[v][u]
            for (int u = 0; u < 8; ++u)
                for (int v = 0; v < 8; ++v)
                    output[(by + u) * width + (bx + v)] = tr[v][u];
        }
    }
}

// ------------------------------------------------------------
// Kernel 0: naive direct 2D DCT (double sum), global reads/writes
// O(N^2) per coefficient, total O(N^4) per 8x8 block if on CPU
// +: No warp divergence, 
// -: Many redundant global memory accesses ---> low arithmetic intensity
// -: Not memory coalesced, since dct on 8x8 tile, each 8 elements in a row is contiguous, but between rows is stride of 'width'.
// ------------------------------------------------------------
__global__ void dct8x8_gpu_naive(const float* __restrict__ input,
                              float* __restrict__ output,
                              int width, int height)
{
    const int tile_x0 = blockIdx.x * 8;
    const int tile_y0 = blockIdx.y * 8;

    const int v = threadIdx.x; // 0..7
    const int u = threadIdx.y; // 0..7

    if (tile_x0 + 7 >= width || tile_y0 + 7 >= height) return;

    float sum = 0.0f;
    #pragma unroll
    for (int y = 0; y < 8; ++y) {
        const float cu = COS8(u, y);
        const int gy = tile_y0 + y;
        #pragma unroll
        for (int x = 0; x < 8; ++x) {
            const float cv = COS8(v, x);
            const int gx = tile_x0 + x;
            const float pix = input[gy * width + gx];
            sum += pix * cu * cv;
        }
    }

    output[(tile_y0 + u) * width + (tile_x0 + v)] = sum;
}

// ------------------------------------------------------------
// Kernel 1: naive direct 2D DCT (double sum) but use shared memory
// only difference from kernel 0 is loading tile into shared memory first
// O(N^2) per coefficient, total O(N^4) per 8x8 block if on CPU
// +: higher arithmetic intensity than kernel 0
// ------------------------------------------------------------
__global__ void dct8x8_gpu_naive_shared_memory(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int width, int height)
{
    const int tile_x0 = (int)blockIdx.x * 8;
    const int tile_y0 = (int)blockIdx.y * 8;

    const int v = (int)threadIdx.x; // 0..7
    const int u = (int)threadIdx.y; // 0..7

    if (tile_x0 + 7 >= width || tile_y0 + 7 >= height) return;

    __shared__ float s_in[8][8 + 1];
    s_in[u][v] = input[(tile_y0 + u) * width + (tile_x0 + v)];
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll // unroll for loop, I think may help perf. https://docs.nvidia.com/cuda/cuda-c-programming-guide/#pragma-unroll
    for (int y = 0; y < 8; ++y) {
        const float cu = COS8(u, y);
        #pragma unroll
        for (int x = 0; x < 8; ++x) {
            const float cv = COS8(v, x);
            sum += s_in[y][x] * cu * cv;
        }
    }

    output[(tile_y0 + u) * width + (tile_x0 + v)] = sum;
}

// ------------------------------------------------------------
// Kernel 2: Naive separable DCT using GLOBAL scratch (no shared memory)
// Before I write this kernel, I expect its performance to be worse than kernel 3 (shared memory version).
// I just want to see how it performs.
//
// The reason there is a scratch buffer which is same size as input image:
// 1. DCT separable needs intermediate storage between row DCT and column DCT. 
// All threads execute DCT parallely, we cannot assume one thread finish execution before the others. 
// So we cannot use a local variable such as tmp[8][8] as I did in cpu version.
//
// -: using global memory for scratch is slow. __syncthreads for each block.
// ------------------------------------------------------------
__global__ void dct8x8_gpu_naive_separable(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       float* __restrict__ scratch, // same size as image
                                       int width, int height)
{
    const int tile_x0 = blockIdx.x * 8;
    const int tile_y0 = blockIdx.y * 8;

    const int v = threadIdx.x; // 0..7
    const int u = threadIdx.y; // 0..7

    if (tile_x0 + 7 >= width || tile_y0 + 7 >= height) return;

    // Phase A: row transform for row=(tile_y0+u)
    float row_sum = 0.0f;
    const int gy = tile_y0 + u;
    #pragma unroll
    for (int x = 0; x < 8; ++x) {
        row_sum += input[gy * width + (tile_x0 + x)] * COS8(v, x);
    }
    scratch[(tile_y0 + u) * width + (tile_x0 + v)] = row_sum;

    // column dct based on the result of row dct
    __syncthreads();

    // Phase B: column transform
    float col_sum = 0.0f;
    #pragma unroll
    for (int y = 0; y < 8; ++y) {
        col_sum += scratch[(tile_y0 + y) * width + (tile_x0 + v)] * COS8(u, y);
    }
    output[(tile_y0 + u) * width + (tile_x0 + v)] = col_sum;
}


// ------------------------------------------------------------
// Kernel 3: Separable DCT using shared memory + shared transpose
// NO scaling factors.
// ------------------------------------------------------------
__global__ void dct8x8_gpu_naive_separable_shared_memory(const float* __restrict__ input,
                                                     float* __restrict__ output,
                                                     int width, int height)
{
    const int tile_x0 = (int)blockIdx.x * 8;
    const int tile_y0 = (int)blockIdx.y * 8;

    const int v = (int)threadIdx.x; // 0..7
    const int u = (int)threadIdx.y; // 0..7

    if (tile_x0 + 7 >= width || tile_y0 + 7 >= height) return;

    // static shared memory
    __shared__ float s_in[8][8]; // __shared__ float s_in[8][8+1]; interestingly, if padding + 1 used, no shared load bank conflict showed by ncu, I digged into this. But have no idea why.
    __shared__ float s_tr[8][8]; // __shared__ float s_tr[8][8+1];

    s_in[u][v] = input[(tile_y0 + u) * width + (tile_x0 + v)];
    __syncthreads();

    float row_sum = 0.0f;
    #pragma unroll
    for (int x = 0; x < 8; ++x) {
        // load
        row_sum += s_in[u][x] * COS8(v, x);
    }

    s_tr[v][u] = row_sum;
    __syncthreads();

    float col_sum = 0.0f;
    #pragma unroll
    for (int y = 0; y < 8; ++y) {
        // load
        col_sum += s_tr[v][y] * COS8(u, y); // __shared__ float s_in/s_tr[8][8+1]; interestingly, if padding + 1 used, no shared load bank conflict showed by ncu, I digged into this. But have no idea why. 
    }

    output[(tile_y0 + u) * width + (tile_x0 + v)] = col_sum;
}

// helper function for Kernel 4
__device__ __forceinline__
void lee8_dct1d_inplace(float (*&curr)[9], float (*&next)[9], int vec, int tid)
{
    // Forward: len=8, half=4, cosOffset=0
    if (tid < 4) {
        float a = curr[vec][tid];
        float b = curr[vec][7 - tid];
        next[vec][tid]     = a + b;
        next[vec][tid + 4] = (a - b) * LEE8(tid);   // c0..c3
    }
    __syncthreads();
    { auto tmp = curr; curr = next; next = tmp; }

    // Forward: len=4, half=2, cosOffset=4, applied to two segments (base=0 and base=4)
    if (tid < 4) {
        int i    = tid & 1;              // 0 or 1
        int base = (tid - i) * 2;        // 0,0,4,4
        float a = curr[vec][base + i];
        float b = curr[vec][base + (3 - i)];
        next[vec][base + i]     = a + b;
        next[vec][base + i + 2] = (a - b) * LEE8(4 + i); // c4,c5
    }
    __syncthreads();
    { auto tmp = curr; curr = next; next = tmp; }

    // Forward: len=2, half=1, cosOffset=6, applied to 4 segments (base=0,2,4,6)
    if (tid < 4) {
        int base = tid * 2;
        float a = curr[vec][base + 0];
        float b = curr[vec][base + 1];
        next[vec][base + 0] = a + b;
        next[vec][base + 1] = (a - b) * LEE8(6);
    }
    __syncthreads();
    { auto tmp = curr; curr = next; next = tmp; }

    // Backward: len=4, half=2
    if (tid < 4) {
        int i    = tid & 1;
        int base = (tid - i) * 2;        // 0,0,4,4
        int out  = base + i * 2;         // 0,2,4,6

        next[vec][out + 0] = curr[vec][base + i];
        next[vec][out + 1] = (i == 1)
            ? curr[vec][base + 3]
            : (curr[vec][base + 2] + curr[vec][base + 3]);
    }
    __syncthreads();
    { auto tmp = curr; curr = next; next = tmp; }

    // Backward: len=8, half=4
    if (tid < 4) {
        int i   = tid;       // 0..3
        int out = i * 2;     // 0,2,4,6

        next[vec][out + 0] = curr[vec][i];
        next[vec][out + 1] = (i == 3)
            ? curr[vec][7]
            : (curr[vec][4 + i] + curr[vec][4 + i + 1]);
    }
    __syncthreads();
    { auto tmp = curr; curr = next; next = tmp; }
}


// ------------------------------------------------------------
// Kernel 4: Separable DCT using shared memory + shared transpose + Lee's algorithm
// ------------------------------------------------------------
__global__ void dct8x8_gpu_lee8_shared(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            int width, int height)
{
    const int tile_x0 = (int)blockIdx.x * 8;
    const int tile_y0 = (int)blockIdx.y * 8;

    const int v = (int)threadIdx.x; // 0..7
    const int u = (int)threadIdx.y; // 0..7

    __shared__ float s0[8][8 + 1]; // +1 to avoid shared memory bank conflict on load
    __shared__ float s1[8][8 + 1];

    // curr is a pointer that points to array of 9 floats
    float (*curr)[9] = s0;
    float (*next)[9] = s1;

    // load tile
    curr[u][v] = input[(tile_y0 + u) * width + (tile_x0 + v)];
    __syncthreads();

    // 1) row-wise Lee DCT: vec=u, tid=v
    lee8_dct1d_inplace(curr, next, /*vec=*/u, /*tid=*/v);

    // 2) transpose: next[v][u] = curr[u][v]
    next[v][u] = curr[u][v];
    __syncthreads();
    { auto tmp = curr; curr = next; next = tmp; }

    // 3) col-wise Lee DCT (on transposed): vec=v, tid=u
    lee8_dct1d_inplace(curr, next, /*vec=*/v, /*tid=*/u);

    // curr[v][u] == coefficient (u,v), here we write back the transposed result
    output[(tile_y0 + u) * width + (tile_x0 + v)] = curr[v][u];
}

#ifndef DCT8X8_DISABLE_MAIN
int main(int argc, char** argv)
{
    int kernel_to_run = 0;
    if (argc >= 2) {
        char* end = nullptr;
        long v = std::strtol(argv[1], &end, 10);
        if (end == argv[1] || (end && *end != '\0') || v < 0 || v > 4) {
            std::fprintf(stderr, "Usage: %s [kernel_id 0-4]\n", argv[0]);
            return 1;
        }
        kernel_to_run = (int)v;
    }

    constexpr int H = 1024;
    constexpr int W = 1024;
    constexpr int TILE = 8;
    static_assert((W % TILE) == 0 && (H % TILE) == 0, "W/H must be divisible by 8");

    init_cos_table_8_gpu();
    init_lee_table_8_gpu();

    const size_t numel = (size_t)W * (size_t)H;
    std::vector<float> h_in(numel);
    std::vector<float> h_out_gpu(numel, 0.0f);
    std::vector<float> h_out_cpu_ref(numel, 0.0f);

    // make input random between -128 .. 127
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-128.0f, 127.0f);
    for (size_t i = 0; i < numel; ++i) h_in[i] = dist(rng);

    using clock = std::chrono::high_resolution_clock;

    // ---------------- CPU naive 2D (double-sum reference) ----------------
    auto t0 = clock::now();
    dct8x8_cpu_naive_reference(h_in.data(), h_out_cpu_ref.data(), W, H);
    auto t1 = clock::now();
    double cpu_ref_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---------------- CPU separable (row 1D + col 1D) ----------------
    std::vector<float> h_out_cpu_sep(numel, 0.0f);
    auto ts0 = clock::now();
    dct8x8_cpu_separable(h_in.data(), h_out_cpu_sep.data(), W, H);
    auto ts1 = clock::now();
    double cpu_sep_ms = std::chrono::duration<double, std::milli>(ts1 - ts0).count();

    // ---------------- run one GPU kernel ----------------
    float *d_in = nullptr, *d_out = nullptr, *d_scratch = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in,  numel * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, numel * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), numel * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0, numel * sizeof(float)));

    if (kernel_to_run == 2) {
        CHECK_CUDA(cudaMalloc(&d_scratch, numel * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_scratch, 0, numel * sizeof(float)));
    }

    dim3 block(8, 8, 1);
    dim3 grid(W / 8, H / 8, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    switch (kernel_to_run) {
        case 0:
            dct8x8_gpu_naive<<<grid, block>>>(d_in, d_out, W, H);
            break;
        case 1:
            dct8x8_gpu_naive_shared_memory<<<grid, block>>>(d_in, d_out, W, H);
            break;
        case 2:
            dct8x8_gpu_naive_separable<<<grid, block>>>(d_in, d_out, d_scratch, W, H);
            break;
        case 3:
            dct8x8_gpu_naive_separable_shared_memory<<<grid, block>>>(d_in, d_out, W, H);
            break;
        case 4:
            dct8x8_gpu_lee8_shared<<<grid, block>>>(d_in, d_out, W, H);
            break;
        default:
            break;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out_gpu.data(), d_out, numel * sizeof(float), cudaMemcpyDeviceToHost));

    // if macro RUN_CPU_LEE8 defined, run CPU Lee version and compare with CPU reference
#if RUN_CPU_LEE8
    std::vector<float> h_out_cpu_lee(numel, 0.0f);

    auto t2 = clock::now();
    dct8x8_cpu_lee(h_in.data(), h_out_cpu_lee.data(), W, H);
    auto t3 = clock::now();
    double cpu_lee_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    printf("------------------------------- CPU naive vs CPU Lee --------------------------------\n");
    printf("CPU naive (2D double-sum) time : %.3f ms\n", cpu_ref_ms);
    printf("CPU Lee (separable + Lee 1D)   : %.3f ms\n", cpu_lee_ms);

    const float cpu_atol = 2e-3f;
    const float cpu_rtol = 1e-6f;
    compare_arrays_and_print("CPU-Lee vs CPU-Ref",
                             h_out_cpu_lee.data(), h_out_cpu_ref.data(),
                             numel, W, cpu_atol, cpu_rtol);
#endif

    // ---------------- CPU separable vs CPU reference ----------------
    printf("------------------------------- CPU naive vs CPU separable --------------------------------\n");
    printf("CPU naive  (2D double-sum) time : %.3f ms\n", cpu_ref_ms);
    printf("CPU sep.   (row+col 1D DCT)     : %.3f ms\n", cpu_sep_ms);

    const float sep_atol = 2e-3f;
    const float sep_rtol = 1e-6f;
    compare_arrays_and_print("CPU-Sep vs CPU-Ref",
                             h_out_cpu_sep.data(), h_out_cpu_ref.data(),
                             numel, W, sep_atol, sep_rtol);

    // ---------------- GPU vs CPU reference ----------------
    printf("----------------------------- CPU vs GPU -----------------------------\n");
    printf("GPU kernel time               : %.3f ms\n", gpu_ms);
    printf("CPU DCT (naive 2D)            : %.3f ms\n", cpu_ref_ms);
    printf("CPU DCT (separable row+col)   : %.3f ms\n", cpu_sep_ms);
#if RUN_CPU_LEE8
    printf("CPU DCT (Lee)                 : %.3f ms\n", cpu_lee_ms);
#endif

    const float gpu_atol = 2e-3f;
    const float gpu_rtol = 1e-6f;

    bool ok_gpu = compare_arrays_and_print("GPU vs CPU-Ref",
                                           h_out_gpu.data(), h_out_cpu_ref.data(),
                                           numel, W, gpu_atol, gpu_rtol);

    if (ok_gpu) {
        printf("[OK] GPU matches CPU-Ref under abs <= atol + rtol * |b|\n");
    } else {
        printf("[FAIL] GPU differs from CPU-Ref under abs <= atol + rtol * |b|\n");
    }

    // --------------------- print some outputs, GPU kernel and CPU reference ----------------
    printf("DCT_KERNEL_TO_RUN = %d\n", kernel_to_run);
    printf("First block (u=0, v=0..7): GPU:\n");
    for (int v = 0; v < 8; ++v) {
        printf("% .6f ", h_out_gpu[0 * W + v]);
    }
    printf("\nFirst block (u=0, v=0..7): CPU-Ref:\n");
    for (int v = 0; v < 8; ++v) {
        printf("% .6f ", h_out_cpu_ref[0 * W + v]);
    }
    printf("\n");

    // ---------------- cleanup ----------------
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    if (d_scratch) CHECK_CUDA(cudaFree(d_scratch));

    return ok_gpu ? 0 : 1;
}
#endif
