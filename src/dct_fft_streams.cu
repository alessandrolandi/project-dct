#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>

#define PI 3.14159265358979323846
#define TILE_DIM 32
#define NUM_STREAMS 4 

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
}

#define CUFFT_CHECK(call) { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) { \
        printf("cuFFT Error: %d at %s:%d\n", err, __FILE__, __LINE__); \
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


// Reorder (Makhoul's method)
__global__ void reorder_rows_kernel(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int src_idx;
        if (x < width/2) {
            src_idx = 2 * x;
        } else {
            src_idx = 2 * (width - 1 - x) + 1;
        }
        output[y * width + x] = input[y * width + src_idx];
    }
}

// Post-process (Apply Twiddle Factors)
__global__ void post_process_rows_kernel(const cufftComplex* fft_output, float* dct_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        cufftComplex Zk;
        int row_offset_complex = y * (width / 2 + 1);
        
        if (x <= width / 2) {
            Zk = fft_output[row_offset_complex + x];
        } else {
            int sym_x = width - x;
            cufftComplex Z_sym = fft_output[row_offset_complex + sym_x];
            Zk.x = Z_sym.x;
            Zk.y = -Z_sym.y; 
        }

        float angle = PI * x / (2.0f * width);
        float cos_val = __cosf(angle);
        float sin_val = __sinf(angle);
        
        float real_part = Zk.x * cos_val + Zk.y * sin_val;
        
        float scale = (x == 0) ? sqrtf(1.0f / width) : sqrtf(2.0f / width);
        dct_output[y * width + x] = real_part * scale;
    }
}


__global__ void transpose_offset_kernel(const float* input, float* output, 
                                        int width, int chunk_height, 
                                        int out_stride, int y_offset) {
    __shared__ float tile[TILE_DIM][TILE_DIM+1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x; 
    int y = blockIdx.y * TILE_DIM + threadIdx.y; 

    if (x < width && y < chunk_height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    __syncthreads();


    
    int x_out = blockIdx.y * TILE_DIM + threadIdx.x; 
    int y_out = blockIdx.x * TILE_DIM + threadIdx.y; 

    if (x_out < chunk_height && y_out < width) {
        int global_col = y_offset + x_out; 
        int global_row = y_out;            
        
        output[global_row * out_stride + global_col] = tile[threadIdx.x][threadIdx.y];
    }
}

void dct2d_fft_streaming(const float* h_input, float* h_output, int rows, int cols) {
    size_t num_pixels = rows * cols;
    size_t bytes_real = num_pixels * sizeof(float);

    float *d_input, *d_reordered_row, *d_transposed;
    float *d_reordered_col; 
    cufftComplex *d_fft_row_out, *d_fft_col_out;

    CUDA_CHECK(cudaMalloc(&d_input, bytes_real));
    CUDA_CHECK(cudaMalloc(&d_transposed, bytes_real)); 
    
    int chunk_rows = (rows + NUM_STREAMS - 1) / NUM_STREAMS;
    int chunk_cols = (cols + NUM_STREAMS - 1) / NUM_STREAMS; // For second pass
    
    size_t max_chunk_elems = (chunk_rows > chunk_cols ? chunk_rows : chunk_cols) * (rows > cols ? rows : cols);
    
    float* d_scratch_reorder[NUM_STREAMS];
    cufftComplex* d_scratch_fft[NUM_STREAMS];
    cudaStream_t streams[NUM_STREAMS];
    cufftHandle plan_rows[NUM_STREAMS];
    cufftHandle plan_cols[NUM_STREAMS];

    for(int i=0; i<NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaMalloc(&d_scratch_reorder[i], max_chunk_elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scratch_fft[i], max_chunk_elems * sizeof(cufftComplex))); // Overkill but safe
        
   
        if (cufftPlan1d(&plan_rows[i], cols, CUFFT_R2C, chunk_rows) != CUFFT_SUCCESS) {
        }
        cufftSetStream(plan_rows[i], streams[i]);

        if (cufftPlan1d(&plan_cols[i], rows, CUFFT_R2C, chunk_cols) != CUFFT_SUCCESS) {
        }
        cufftSetStream(plan_cols[i], streams[i]);
    }

    GpuTimer timer;
    printf("Running Streamed 2D DCT (%dx%d) with %d streams...\n", cols, rows, NUM_STREAMS);
    timer.Start();

    for (int i = 0; i < NUM_STREAMS; i++) {
        int start_row = i * chunk_rows;
        int current_rows = (start_row + chunk_rows > rows) ? (rows - start_row) : chunk_rows;
        if (current_rows <= 0) break;

        size_t offset_elems = start_row * cols;
        float* d_chunk_in = d_input + offset_elems;
        
        CUDA_CHECK(cudaMemcpyAsync(d_chunk_in, h_input + offset_elems, 
                                   current_rows * cols * sizeof(float), 
                                   cudaMemcpyHostToDevice, streams[i]));

        dim3 block(16, 16);
        dim3 grid((cols + 15)/16, (current_rows + 15)/16);
        reorder_rows_kernel<<<grid, block, 0, streams[i]>>>(d_chunk_in, d_scratch_reorder[i], cols, current_rows);

        CUFFT_CHECK(cufftExecR2C(plan_rows[i], d_scratch_reorder[i], d_scratch_fft[i]));

        post_process_rows_kernel<<<grid, block, 0, streams[i]>>>(d_scratch_fft[i], d_chunk_in, cols, current_rows);

        dim3 gridTrans((cols + TILE_DIM - 1)/TILE_DIM, (current_rows + TILE_DIM - 1)/TILE_DIM);
        transpose_offset_kernel<<<gridTrans, dim3(TILE_DIM, TILE_DIM), 0, streams[i]>>>(
            d_chunk_in, d_transposed, 
            cols, current_rows, 
            rows, start_row // stride is 'rows' because transposed matrix is Cols x Rows
        );
    }

    // barroer
    for(int i=0; i<NUM_STREAMS; i++) cudaStreamSynchronize(streams[i]);

    // Col FFT (on transposed data) -> Transpose Back -> D2H

    for (int i = 0; i < NUM_STREAMS; i++) {
        int start_col = i * chunk_cols; // These are rows in the transposed matrix
        int current_cols = (start_col + chunk_cols > cols) ? (cols - start_col) : chunk_cols;
        if (current_cols <= 0) break;

        size_t offset_elems = start_col * rows;
        float* d_chunk_trans = d_transposed + offset_elems;
        
        // 1. Reorder (Chunk -> Scratch)
        dim3 block(16, 16);
        dim3 grid((rows + 15)/16, (current_cols + 15)/16);
        reorder_rows_kernel<<<grid, block, 0, streams[i]>>>(d_chunk_trans, d_scratch_reorder[i], rows, current_cols);

        CUFFT_CHECK(cufftExecR2C(plan_cols[i], d_scratch_reorder[i], d_scratch_fft[i]));

        post_process_rows_kernel<<<grid, block, 0, streams[i]>>>(d_scratch_fft[i], d_chunk_trans, rows, current_cols);

        dim3 gridTrans((rows + TILE_DIM - 1)/TILE_DIM, (current_cols + TILE_DIM - 1)/TILE_DIM);
        transpose_offset_kernel<<<gridTrans, dim3(TILE_DIM, TILE_DIM), 0, streams[i]>>>(
            d_chunk_trans, d_input, 
            rows, current_cols, 
            cols, start_col
        );

    }

    for(int i=0; i<NUM_STREAMS; i++) cudaStreamSynchronize(streams[i]);
    
    CUDA_CHECK(cudaMemcpyAsync(h_output, d_input, bytes_real, cudaMemcpyDeviceToHost, streams[0]));
    cudaStreamSynchronize(streams[0]);

    timer.Stop();
    printf("DCT via FFT Time: %.3f ms\n", timer.Elapsed());

    for(int i=0; i<NUM_STREAMS; i++) {
        cufftDestroy(plan_rows[i]);
        cufftDestroy(plan_cols[i]);
        cudaFree(d_scratch_reorder[i]);
        cudaFree(d_scratch_fft[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_input);
    cudaFree(d_transposed);
}


int main() {
    printf(" Global 2D DCT via Streamed FFT \n\n");

    auto alloc_pinned = [](size_t n) -> float* {
        float* ptr;
        cudaMallocHost(&ptr, n * sizeof(float));
        return ptr;
    };
    auto free_pinned = [](float* ptr) {
        cudaFreeHost(ptr);
    };

    // Test 1: 8x8 Reference
    const int N1 = 8;
    float* input1 = alloc_pinned(64);
    float* output1 = alloc_pinned(64);
    for(int i=0; i<64; i++) input1[i] = i + 1;

    printf("Test 1: Single 8x8 block\n");
    dct2d_fft_streaming(input1, output1, N1, N1);
    
    printf("DCT (Top-Left 8x8):\n");
    for(int y=0; y<8; y++) {
        for(int x=0; x<8; x++) printf("%8.2f ", output1[y*8+x]);
        printf("\n");
    }
    printf("\n");
    free_pinned(input1); free_pinned(output1);

    // Test 2: 300x300
    const int W2 = 300, H2 = 300;
    float* input2 = alloc_pinned(W2*H2);
    float* output2 = alloc_pinned(W2*H2);
    for(int i=0; i<W2*H2; i++) input2[i] = (float)((i % 128) + 1);

    printf("Test 2: 300x300 image\n");
    dct2d_fft_streaming(input2, output2, H2, W2);

    printf("DCT (Top-Left 4x4):\n");
    for(int y=0; y<4; y++) {
        for(int x=0; x<4; x++) printf("%8.2f ", output2[y*W2+x]);
        printf("\n");
    }
    printf("\n");
    free_pinned(input2); free_pinned(output2);

    // Test 3: 2048x2048
    const int W3 = 2048, H3 = 2048;
    float* input3 = alloc_pinned(W3*H3);
    float* output3 = alloc_pinned(W3*H3);
    for(int i=0; i<W3*H3; i++) input3[i] = (float)((i % 128) + 1);

    printf("Test 3: 2048x2048 image (Benchmark)\n");
    dct2d_fft_streaming(input3, output3, H3, W3);
    
    printf("DCT (Top-Left 4x4):\n");
    for(int y=0; y<4; y++) {
        for(int x=0; x<4; x++) printf("%8.2f ", output3[y*W3+x]);
        printf("\n");
    }
    printf("\n");
    free_pinned(input3); free_pinned(output3);

    // Test 4: 4096x4096 
    const int W4 = 4096, H4 = 4096;
    float* input4 = alloc_pinned(W4*H4);
    float* output4 = alloc_pinned(W4*H4);
    for(int i=0; i<W4*H4; i++) input4[i] = (float)((i % 128) + 1);

    printf("Test 4: 4096x4096 image\n");
    dct2d_fft_streaming(input4, output4, H4, W4);
    
    printf("DCT (Top-Left 4x4):\n");
    for(int y=0; y<4; y++) {
        for(int x=0; x<4; x++) printf("%8.2f ", output4[y*W4+x]);
        printf("\n");
    }
    printf("\n");
    free_pinned(input4); free_pinned(output4);

    return 0;
}