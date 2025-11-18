#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>

#define PI 3.1415
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

// Method 1: Even Extension (more simple)

// Kernel to create even symmetric extension
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
        float scale = (k == 0) ? sqrtf(1.0f / (2*N)) : sqrtf(2.0f / (2*N));
        // DCT is the real part of FFT, scaled appropriately
        dct_output[k] = fft_result[k].x * scale * 2.0f;
    }
}

void dct1d_fft_method1(const float* h_input, float* h_output, int N) {
    float *d_input, *d_extended, *d_output;
    cufftComplex *d_fft_result;
    
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_extended, 2*N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fft_result, (2*N) * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Step 1: Create even symmetric extension
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    even_extend_kernel<<<gridSize, blockSize>>>(d_input, d_extended, N);
    CUDA_CHECK(cudaGetLastError());
    
    // Step 2: Compute FFT on extended signal
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, 2*N, CUFFT_R2C, 1));
    CUFFT_CHECK(cufftExecR2C(plan, d_extended, d_fft_result));
    
    // Step 3: Extract DCT from FFT result
    extract_dct_from_fft<<<gridSize, blockSize>>>(d_fft_result, d_output, N);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_input);
    cudaFree(d_extended);
    cudaFree(d_fft_result);
    cudaFree(d_output);
}

// Methoid 2: Reordering (More efficient - uses N-point FFT instead of 2N)

// Kernel to reorder input: even indices first, then odd indices reversed
__global__ void reorder_for_fft(const float* input, float* reordered, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        if (idx < N/2) {
            // First half: even indices (0, 2, 4, ...)
            reordered[idx] = input[2*idx];
        } else {
            // Second half: odd indices reversed (N-1, N-3, N-5, ...)
            int odd_idx = 2 * (N - 1 - idx) + 1;
            reordered[idx] = input[odd_idx];
        }
    }
}

// Kernel to apply twiddle factors and extract DCT
__global__ void apply_twiddles_and_extract(const cufftComplex* fft_result, 
                                            float* dct_output, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k < N) {
        // Twiddle factor: e^(-i*pi*k/(2N)) = cos(pi*k/(2N)) - i*sin(pi*k/(2N))
        float angle = PI * k / (2.0f * N);
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        
        // Multiply FFT result by twiddle factor
        // (a + bi) * (c - di) = (ac + bd) + (bc - ad)i
        float real_part = fft_result[k].x * cos_val + fft_result[k].y * sin_val;
        
        // Scale for DCT normalization
        float scale = (k == 0) ? sqrtf(1.0f / N) : sqrtf(2.0f / N);
        dct_output[k] = real_part * scale;
    }
}

void dct1d_fft_method2(const float* h_input, float* h_output, int N) {
    float *d_input, *d_reordered, *d_output;
    cufftComplex *d_fft_result;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_reordered, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fft_result, (N/2 + 1) * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Step 1: Reorder input
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    reorder_for_fft<<<gridSize, blockSize>>>(d_input, d_reordered, N);
    CUDA_CHECK(cudaGetLastError());
    
    // Step 2: Compute N-point FFT (Real to Complex)
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_R2C, 1));
    CUFFT_CHECK(cufftExecR2C(plan, d_reordered, d_fft_result));
    
    // Step 3: Apply twiddle factors and extract DCT
    apply_twiddles_and_extract<<<gridSize, blockSize>>>(d_fft_result, d_output, N);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Cleanup
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
    
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_temp, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    // Process rows: Apply 1D DCT to each row
    for (int row = 0; row < rows; row++) {
        float *row_input = d_input + row * cols;
        float *row_output = d_temp + row * cols;
        
        // Use in-place FFT method for each row
        float *d_row_reordered;
        cufftComplex *d_row_fft;
        
        CUDA_CHECK(cudaMalloc(&d_row_reordered, cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_row_fft, (cols/2 + 1) * sizeof(cufftComplex)));
        
        reorder_for_fft<<<1, (cols + 255) / 256 * 256>>>(row_input, d_row_reordered, cols);
        
        cufftHandle plan_row;
        CUFFT_CHECK(cufftPlan1d(&plan_row, cols, CUFFT_R2C, 1));
        CUFFT_CHECK(cufftExecR2C(plan_row, d_row_reordered, d_row_fft));
        
        apply_twiddles_and_extract<<<1, (cols + 255) / 256 * 256>>>(d_row_fft, row_output, cols);
        
        cufftDestroy(plan_row);
        cudaFree(d_row_reordered);
        cudaFree(d_row_fft);
    }
    
    // Process columns: Apply 1D DCT to each column
    // Note: This is inefficient (non-coalesced access); will have to change it later
    for (int col = 0; col < cols; col++) {
        float *h_col = (float*)malloc(rows * sizeof(float));
        float *h_col_out = (float*)malloc(rows * sizeof(float));
        
        // Extract column
        for (int row = 0; row < rows; row++) {
            CUDA_CHECK(cudaMemcpy(&h_col[row], d_temp + row * cols + col, 
                                 sizeof(float), cudaMemcpyDeviceToHost));
        }
        
        // Apply 1D DCT to column
        dct1d_fft_method2(h_col, h_col_out, rows);
        
        // Put column back
        for (int row = 0; row < rows; row++) {
            CUDA_CHECK(cudaMemcpy(d_output + row * cols + col, &h_col_out[row], 
                                 sizeof(float), cudaMemcpyHostToDevice));
        }
        
        free(h_col);
        free(h_col_out);
    }
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
}

// Test and comparison

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
    
    // Test 1: Small 1D DCT
    const int N = 8;
    float input[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float output_ref[N];
    float output_fft1[N];
    float output_fft2[N];
    
    printf("Input signal: ");
    for (int i = 0; i < N; i++) printf("%.1f ", input[i]);
    printf("\n\n");
    
    // Reference direct DCT
    dct1d_reference(input, output_ref, N);
    printf("Reference DCT (direct):\n");
    for (int i = 0; i < N; i++) printf("%.4f ", output_ref[i]);
    printf("\n\n");
    
    // Method 1: Even extension
    dct1d_fft_method1(input, output_fft1, N);
    printf("DCT via FFT (Method 1 - Even Extension):\n");
    for (int i = 0; i < N; i++) printf("%.4f ", output_fft1[i]);
    printf("\n\n");
    
    // Method 2: Reordering
    dct1d_fft_method2(input, output_fft2, N);
    printf("DCT via FFT (Method 2 - Reordering):\n");
    for (int i = 0; i < N; i++) printf("%.4f ", output_fft2[i]);
    printf("\n\n");
    
    // Compare errors
    float max_error1 = 0.0f, max_error2 = 0.0f;
    for (int i = 0; i < N; i++) {
        float err1 = fabsf(output_ref[i] - output_fft1[i]);
        float err2 = fabsf(output_ref[i] - output_fft2[i]);
        if (err1 > max_error1) max_error1 = err1;
        if (err2 > max_error2) max_error2 = err2;
    }
    
    printf("Maximum error (Method 1): %.2e\n", max_error1);
    printf("Maximum error (Method 2): %.2e\n", max_error2);
    printf("\n");
    
    // Test 2: 2D DCT
    printf("2D DCT Test\n");
    const int rows = 4, cols = 4;
    float input2d[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float output2d[16];
    
    printf("Input 2D:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.0f ", input2d[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    dct2d_fft(input2d, output2d, rows, cols);
    
    printf("2D DCT via FFT:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.2f ", output2d[i * cols + j]);
        }
        printf("\n");
    }
    
    printf("\nTest Complete\n");
    
    return 0;
}