#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <math.h>

#define PI 3.14159265358979323846
#define TILE_WIDTH 16
#define NUM_STREAMS 8
#define THRESHOLD 0.1f // Values below this are treated as 0

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
}

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void Start() { cudaEventRecord(start, 0); }
    void Stop() { cudaEventRecord(stop, 0); }
    float Elapsed() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};



__nv_bfloat16 float_to_bf16_cpu(float f) {
    unsigned int* f_bits = (unsigned int*)&f;
    unsigned short bf_bits = (unsigned short)((*f_bits) >> 16);
    return *(__nv_bfloat16*)&bf_bits;
}


__global__ void generate_dct_matrix_kernel_fp32(float* matrix, int N, bool transpose) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x >= N || idx_y >= N) return;

    int k, n;
    if (transpose) { n = idx_y; k = idx_x; } 
    else { k = idx_y; n = idx_x; }

    float arg = PI * k * (2.0f * n + 1.0f) / (2.0f * N);
    float val = __cosf(arg);
    float scale = (k == 0) ? sqrtf(1.0f / N) : sqrtf(2.0f / N);
    
    matrix[idx_y * N + idx_x] = val * scale;
}

__global__ void row_transform_dense(const __nv_bfloat16* A, 
                                    const float* B, 
                                    float* C, 
                                    int M, int K, int N) {
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int local_row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        __shared__ float As[TILE_WIDTH][TILE_WIDTH];
        __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

        int k_idx = t * TILE_WIDTH + tx;
        if (local_row < M && k_idx < K)
             As[ty][tx] = __bfloat162float(__ldg(&A[local_row * K + k_idx]));
        else As[ty][tx] = 0.0f;

        int k_row = t * TILE_WIDTH + ty;
        if (k_row < K && col < N)
             Bs[ty][tx] = __ldg(&B[k_row * N + col]);
        else Bs[ty][tx] = 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }

    if (local_row < M && col < N) {
        C[local_row * N + col] = sum;
    }
}

__global__ void col_transform_sparse_fused(const float* A, 
                                           const float* B, 
                                           int* out_indices,     // Sparse Output Indices
                                           float* out_values,    // Sparse Output Values
                                           int* global_counter,  // Atomic Counter
                                           int max_capacity,
                                           int M, int K, int N,
                                           int row_offset_global) { // To calculate absolute index
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int local_row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        __shared__ float As[TILE_WIDTH][TILE_WIDTH];
        __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

        int k_idx = t * TILE_WIDTH + tx;
        if (local_row < M && k_idx < K) As[ty][tx] = __ldg(&A[local_row * K + k_idx]);
        else As[ty][tx] = 0.0f;

        int k_row = t * TILE_WIDTH + ty;
        if (k_row < K && col < N) Bs[ty][tx] = __ldg(&B[k_row * N + col]);
        else Bs[ty][tx] = 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }

    //SPARSE PACKING LOGIC (imp)
    if (local_row < M && col < N) {
        bool is_nonzero = (fabsf(sum) > THRESHOLD);

        // 2. Warp Vote (Optimization: Reduce Atomic Contention)
        // 'mask' will have a bit set for every thread in the warp that has a non-zero value
        unsigned int mask = __ballot_sync(0xFFFFFFFF, is_nonzero);

        if (is_nonzero) {
            // How many threads before me (in this warp) have data?
            int lane_id = threadIdx.x % 32;
            int leader_id = __ffs(mask) - 1; // First thread with data
            
            int base_offset = 0;
            
            // Only the leader increments the global counter for the WHOLE warp
            if (lane_id == leader_id) {
                base_offset = atomicAdd(global_counter, __popc(mask));
            }
            
            // Broadcast the base offset to all threads in warp
            base_offset = __shfl_sync(0xFFFFFFFF, base_offset, leader_id);
            
            // Calculate my personal offset
            // __popc(mask & ((1 << lane_id) - 1)) counts set bits below my lane index
            int my_offset = base_offset + __popc(mask & ((1 << lane_id) - 1));

            // 3. Write Data (if within buffer limits)
            if (my_offset < max_capacity) {
                int global_idx = (local_row + row_offset_global) * N + col;
                out_indices[my_offset] = global_idx;
                out_values[my_offset] = sum;
            }
        }
    }
}

void global_dct_sparse(const float* h_input_fp32, float* h_output_dense_host, 
                       int img_height, int img_width) {
    size_t num_pixels = img_width * img_height;
    size_t bytes_bf16 = num_pixels * sizeof(__nv_bfloat16);
    size_t bytes_fp32 = num_pixels * sizeof(float);


    memset(h_output_dense_host, 0, bytes_fp32);

    int active_streams = NUM_STREAMS;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < active_streams; i++) 
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

    int rows_per_stream = (img_height + active_streams - 1) / active_streams;

    __nv_bfloat16 *h_input_bf16;
    CUDA_CHECK(cudaMallocHost(&h_input_bf16, bytes_bf16));
    for(size_t i=0; i<num_pixels; i++) h_input_bf16[i] = float_to_bf16_cpu(h_input_fp32[i]);

    __nv_bfloat16 *d_input;
    float *d_temp, *d_Crow_T, *d_Ccol;
    
    CUDA_CHECK(cudaMalloc(&d_input, bytes_bf16));
    CUDA_CHECK(cudaMalloc(&d_temp, bytes_fp32)); 
    CUDA_CHECK(cudaMalloc(&d_Crow_T, img_width * img_width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ccol, img_height * img_height * sizeof(float)));

    // SPARSE BUFFERS
    int max_sparse_entries = num_pixels * 0.5; 
    int *d_sparse_indices, *d_global_counter;
    float *d_sparse_values;
    
    CUDA_CHECK(cudaMalloc(&d_sparse_indices, max_sparse_entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sparse_values, max_sparse_entries * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_global_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_global_counter, 0, sizeof(int))); // Reset counter

    int* h_sparse_indices;
    float* h_sparse_values;
    int* h_counter_cpu;
    CUDA_CHECK(cudaMallocHost(&h_sparse_indices, max_sparse_entries * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_sparse_values, max_sparse_entries * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_counter_cpu, sizeof(int)));

    // 4. Generate Matrices (FP32)
    dim3 block(16, 16);
    dim3 gridMat((img_width + 15)/16, (img_width + 15)/16);
    generate_dct_matrix_kernel_fp32<<<gridMat, block>>>(d_Crow_T, img_width, true);
    generate_dct_matrix_kernel_fp32<<<gridMat, block>>>(d_Ccol, img_height, false);
    cudaDeviceSynchronize();

    GpuTimer timer;
    printf("Running Sparse DCT (Threshold: %.2f, Warp-Aggregated Atomics)...\n", THRESHOLD);
    timer.Start();

    for (int i = 0; i < active_streams; ++i) {
        int start_row = i * rows_per_stream;
        int num_rows = (start_row + rows_per_stream > img_height) ? 
                       img_height - start_row : rows_per_stream;
        if (num_rows <= 0) break;

        size_t offset_elems = start_row * img_width;
        
        CUDA_CHECK(cudaMemcpyAsync(d_input + offset_elems, h_input_bf16 + offset_elems, 
                                   num_rows * img_width * sizeof(__nv_bfloat16), 
                                   cudaMemcpyHostToDevice, streams[i]));

        dim3 grid((img_width + TILE_WIDTH - 1) / TILE_WIDTH, 
                  (num_rows + TILE_WIDTH - 1) / TILE_WIDTH);
        
        row_transform_dense<<<grid, block, 0, streams[i]>>>(
            d_input + offset_elems, d_Crow_T, d_temp + offset_elems, 
            num_rows, img_width, img_width
        );
    }

    // Barrier before Col Transform
    for (int i = 0; i < active_streams; i++) cudaStreamSynchronize(streams[i]);

    
    for (int i = 0; i < active_streams; ++i) {
        int start_row = i * rows_per_stream;
        int num_rows = (start_row + rows_per_stream > img_height) ? 
                       img_height - start_row : rows_per_stream;
        if (num_rows <= 0) break;

        size_t C_offset_elems = start_row * img_height;
        dim3 grid((img_width + TILE_WIDTH - 1) / TILE_WIDTH, 
                  (num_rows + TILE_WIDTH - 1) / TILE_WIDTH);

        col_transform_sparse_fused<<<grid, block, 0, streams[i]>>>(
            d_Ccol + C_offset_elems, // A chunk
            d_temp,                  // B full
            d_sparse_indices,        // Out Indices
            d_sparse_values,         // Out Values
            d_global_counter,        // Atomic Counter
            max_sparse_entries,      // Safety Limit
            num_rows, img_height, img_width,
            start_row                // Global row offset
        );
    }

    cudaStreamSynchronize(streams[0]); 
    CUDA_CHECK(cudaMemcpy(h_counter_cpu, d_global_counter, sizeof(int), cudaMemcpyDeviceToHost));
    
    int total_non_zeros = *h_counter_cpu;
    if (total_non_zeros > max_sparse_entries) total_non_zeros = max_sparse_entries; // Clamp
    
    printf("  Sparsity Report: %d / %lu pixels (%.2f%% non-zero)\n", 
           total_non_zeros, num_pixels, 100.0f * total_non_zeros / num_pixels);

    
    CUDA_CHECK(cudaMemcpyAsync(h_sparse_indices, d_sparse_indices, 
                               total_non_zeros * sizeof(int), 
                               cudaMemcpyDeviceToHost, streams[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(h_sparse_values, d_sparse_values, 
                               total_non_zeros * sizeof(float), 
                               cudaMemcpyDeviceToHost, streams[0]));

    cudaStreamSynchronize(streams[0]); // Wait for data
    timer.Stop();
    printf("Total GPU + Transfer Time: %.3f ms\n", timer.Elapsed());

    for (int i = 0; i < total_non_zeros; i++) {
        int idx = h_sparse_indices[i];
        float val = h_sparse_values[i];
        h_output_dense_host[idx] = val;
    }

    // Cleanup
    for(int i=0; i<active_streams; i++) cudaStreamDestroy(streams[i]);
    cudaFree(d_input); cudaFree(d_temp); cudaFree(d_Crow_T); cudaFree(d_Ccol);
    cudaFree(d_sparse_indices); cudaFree(d_sparse_values); cudaFree(d_global_counter);
    cudaFreeHost(h_input_bf16); cudaFreeHost(h_sparse_indices); 
    cudaFreeHost(h_sparse_values); cudaFreeHost(h_counter_cpu);
}

int main() {
    printf("=== Sparse Output Optimized DCT ===\n");
    
  
    const int N1 = 8;
    float input1[64];
    for(int i=0; i<64; i++) input1[i] = i + 1;
    float output1[64];

    printf("Test 1: Single 8x8 block\n");
    global_dct_sparse(input1, output1, N1, N1);
    
    printf("DCT (Top-Left 8x8):\n");
    for(int y=0; y<8; y++) {
        for(int x=0; x<8; x++) printf("%8.2f ", output1[y*8+x]);
        printf("\n");
    }
    printf("\n");


    // const int W2 = 200, H2 = 200;
    // int n2 = W2 * H2;
    // float* input2 = new float[n2];
    // float* output2 = new float[n2];
    
    // for(int i=0; i<n2; i++) input2[i] = (float)((i % 128) + 1);

    // printf("Test 2: 300x300 image\n");
    // global_dct_sparse(input2, output2, H2, W2);

    // printf("DCT (Top-Left 4x4):\n");
    // for(int y=0; y<4; y++) {
    //     for(int x=0; x<4; x++) printf("%8.2f ", output2[y*W2+x]);
    //     printf("\n");
    // }
    // printf("\n");

    // delete[] input2; delete[] output2;

    const int W3 = 2048, H3 = 2048;
    int n3 = W3 * H3;
    float* input3 = new float[n3];
    float* output3 = new float[n3];
    for(int i=0; i<n3; i++) input3[i] = (float)((i % 128) + 1);

    printf("Test 3: 2048x2048 image (Benchmark)\n");
    printf("Memory Savings: FP32 would be %.2f MB. Using BF16: %.2f MB\n",
           (n3*4.0)/1024/1024, (n3*2.0)/1024/1024);
    
    global_dct_sparse(input3, output3, H3, W3);
    
    printf("DCT (Top-Left 4x4):\n");
    for(int y=0; y<4; y++) {
        for(int x=0; x<4; x++) printf("%8.2f ", output3[y*W3+x]);
        printf("\n");
    }
    printf("\n");

    delete[] input3; delete[] output3;

    const int W4 = 4096, H4 = 4096;
    int n4 = W4 * H4;
    float* input4 = new float[n4];
    float* output4 = new float[n4];
    for(int i=0; i<n4; i++) input4[i] = (float)((i % 128) + 1);

    printf("Test 4: 4096x4096 image\n");
    global_dct_sparse(input4, output4, H4, W4);
    
    printf("DCT (Top-Left 4x4):\n");
    for(int y=0; y<4; y++) {
        for(int x=0; x<4; x++) printf("%8.2f ", output4[y*W4+x]);
        printf("\n");
    }
    printf("\n");

    delete[] input4; delete[] output4;
    return 0;
}