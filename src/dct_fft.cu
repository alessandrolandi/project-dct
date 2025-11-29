#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>

#define PI 3.14159265358979323846
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
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() {
        cudaEventRecord(start, 0);
    }

    void Stop() {
        cudaEventRecord(stop, 0);
    }

    float Elapsed() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// Method 1: Even Extension (more simple)
__global__ void even_extend_kernel(const float* input, float* extended, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        extended[idx] = input[idx];           // First half: original
        extended[2*N - 1 - idx] = input[idx]; // Second half: reversed
    }
}

// Kernel to extract DCT from FFT result
__global__ void extract_dct_from_fft(const cufftComplex* fft_result, 
                                      float* dct_output, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k < N) {
        float angle = PI * k / (2.0f * N);
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        
        float real_part = fft_result[k].x * cos_val + fft_result[k].y * sin_val;
        
        float ortho_scale = (k == 0) ? sqrtf(1.0f / N) : sqrtf(2.0f / N);
        dct_output[k] = real_part * ortho_scale * 0.5f;
    }
}

void dct1d_fft_method1(const float* h_input, float* h_output, int N) {
    float *d_input, *d_extended, *d_output;
    cufftComplex *d_fft_result;
    GpuTimer timer;
    
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_extended, 2*N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fft_result, (2*N) * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, 2*N, CUFFT_R2C, 1));
    
    // Start Timing
    timer.Start();

    even_extend_kernel<<<gridSize, blockSize>>>(d_input, d_extended, N);
    
    CUFFT_CHECK(cufftExecR2C(plan, d_extended, d_fft_result));
    
    extract_dct_from_fft<<<gridSize, blockSize>>>(d_fft_result, d_output, N);
    
    timer.Stop();
    printf("DCT 1D (Method 1) Time: %.3f ms\n", timer.Elapsed());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    cufftDestroy(plan);
    cudaFree(d_input);
    cudaFree(d_extended);
    cudaFree(d_fft_result);
    cudaFree(d_output);
}

// Method 2: Reordering (More efficient - uses N-point FFT instead of 2N)
__global__ void reorder_for_fft(const float* input, float* reordered, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (idx < N/2) {
            reordered[idx] = input[2*idx];
        } else {
            int odd_idx = 2 * (N - 1 - idx) + 1;
            reordered[idx] = input[odd_idx];
        }
    }
}

__global__ void apply_twiddles_and_extract(const cufftComplex* fft_result, 
                                            float* dct_output, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k < N) {
        cufftComplex Zk;
        if (k <= N / 2) {
            Zk = fft_result[k];
        } else {
            int sym_k = N - k;
            Zk.x = fft_result[sym_k].x;
            Zk.y = -fft_result[sym_k].y; 
        }

        float angle = PI * k / (2.0f * N);
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        
        float real_part = Zk.x * cos_val + Zk.y * sin_val;
        float scale = (k == 0) ? sqrtf(1.0f / N) : sqrtf(2.0f / N);
        dct_output[k] = real_part * scale;
    }
}

void dct1d_fft_method2(const float* h_input, float* h_output, int N) {
    float *d_input, *d_reordered, *d_output;
    cufftComplex *d_fft_result;
    GpuTimer timer;
    
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_reordered, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fft_result, (N/2 + 1) * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_R2C, 1));

    // Start Timing
    timer.Start();

    // Step 1: Reorder input
    reorder_for_fft<<<gridSize, blockSize>>>(d_input, d_reordered, N);
    
    // Step 2: Compute FFT
    CUFFT_CHECK(cufftExecR2C(plan, d_reordered, d_fft_result));
    
    // Step 3: Extract
    apply_twiddles_and_extract<<<gridSize, blockSize>>>(d_fft_result, d_output, N);
    
    timer.Stop();
    printf("DCT 1D (Method 2) Time: %.3f ms\n", timer.Elapsed());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    cufftDestroy(plan);
    cudaFree(d_input);
    cudaFree(d_reordered);
    cudaFree(d_fft_result);
    cudaFree(d_output);
}

// 2D DCT via FFT (using separable property)
void dct2d_fft(const float* h_input, float* h_output, int rows, int cols) {
    size_t size = rows * cols * sizeof(float);
    float *d_input, *d_temp, *d_output;
    GpuTimer timer;
    
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_temp, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    // Start Timing (Includes overhead of CPU-transposes for now)
    timer.Start();

    // Process rows
    for (int row = 0; row < rows; row++) {
        float *row_input = d_input + row * cols;
        float *row_output = d_temp + row * cols;
        float *d_row_reordered;
        cufftComplex *d_row_fft;
        
        CUDA_CHECK(cudaMalloc(&d_row_reordered, cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_row_fft, (cols/2 + 1) * sizeof(cufftComplex)));
        
        reorder_for_fft<<<1, 256>>>(row_input, d_row_reordered, cols);
        
        cufftHandle plan_row;
        CUFFT_CHECK(cufftPlan1d(&plan_row, cols, CUFFT_R2C, 1));
        CUFFT_CHECK(cufftExecR2C(plan_row, d_row_reordered, d_row_fft));
        
        apply_twiddles_and_extract<<<1, 256>>>(d_row_fft, row_output, cols);
        
        cufftDestroy(plan_row);
        cudaFree(d_row_reordered);
        cudaFree(d_row_fft);
    }
    
    // Process columns (Inefficient CPU transpose included in timing)
    for (int col = 0; col < cols; col++) {
        float *h_col = (float*)malloc(rows * sizeof(float));
        float *h_col_out = (float*)malloc(rows * sizeof(float));
        
        for (int row = 0; row < rows; row++) {
            CUDA_CHECK(cudaMemcpy(&h_col[row], d_temp + row * cols + col, 
                                 sizeof(float), cudaMemcpyDeviceToHost));
        }

        dct1d_fft_method2(h_col, h_col_out, rows);
        
        for (int row = 0; row < rows; row++) {
            CUDA_CHECK(cudaMemcpy(d_output + row * cols + col, &h_col_out[row], 
                                 sizeof(float), cudaMemcpyHostToDevice));
        }
        
        free(h_col);
        free(h_col_out);
    }
    
    timer.Stop();
    printf("DCT 2D (FFT-based) Total Time: %.3f ms\n", timer.Elapsed());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
}

// Reference direct DCT for comparison
void dct1d_reference(const float* input, float* output, int N) {
    for (int k = 0; k < N; k++) {
        float sum = 0.0f;
        float scale = (k == 0) ? sqrtf(1.0f / N) : sqrtf(2.0f / N);
        for (int n = 0; n < N; n++) {
            sum += input[n] * cosf(PI * k * (2*n + 1) / (2.0f * N));
        }
        output[k] = scale * sum;
    }
}

int main() {
    printf("DCT via FFT Testing\n\n");
    
    const int N = 8;
    float input[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float output_ref[N], output_fft1[N], output_fft2[N];
    
    printf("Input signal: ");
    for (int i = 0; i < N; i++) printf("%.1f ", input[i]);
    printf("\n\n");
    
    dct1d_reference(input, output_ref, N);
    
    // Method 1
    dct1d_fft_method1(input, output_fft1, N);
    printf("DCT via FFT (Method 1) Result check: %.4f ...\n", output_fft1[0]);
    
    // Method 2
    dct1d_fft_method2(input, output_fft2, N);
    printf("DCT via FFT (Method 2) Result check: %.4f ...\n", output_fft2[0]);
    
    printf("\n");
    
    // Test 2: 2D DCT
    printf("2D DCT Test\n");
    const int rows = 4, cols = 4;
    float input2d[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float output2d[16];
    
    dct2d_fft(input2d, output2d, rows, cols);
    
    printf("2D DCT via FFT:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) printf("%8.2f ", output2d[i * cols + j]);
        printf("\n");
    }
    
    return 0;
}