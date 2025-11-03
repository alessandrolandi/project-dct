#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define PI 3.1415

// 1D DCT-II kernel (basic version)
__global__ void dct1d_kernel(const float* input, float* output, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k < N) {
        float sum = 0.0f;
        float scale = (k == 0) ? sqrtf(1.0f / N) : sqrtf(2.0f / N);
        
        for (int n = 0; n < N; n++) {
            sum += input[n] * cosf(PI * k * (2 * n + 1) / (2.0f * N));
        }
        
        output[k] = scale * sum;
    }
}

// 1D IDCT-II kernel (inverse DCT)
__global__ void idct1d_kernel(const float* input, float* output, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n < N) {
        float sum = 0.0f;
        
        for (int k = 0; k < N; k++) {
            float scale = (k == 0) ? sqrtf(1.0f / N) : sqrtf(2.0f / N);
            sum += scale * input[k] * cosf(PI * k * (2 * n + 1) / (2.0f * N));
        }
        
        output[n] = sum;
    }
}

// 2D DCT using separable property (row-wise then column-wise)
__global__ void dct2d_rows_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && k < cols) {
        float sum = 0.0f;
        float scale = (k == 0) ? sqrtf(1.0f / cols) : sqrtf(2.0f / cols);
        
        for (int n = 0; n < cols; n++) {
            sum += input[row * cols + n] * cosf(PI * k * (2 * n + 1) / (2.0f * cols));
        }
        
        output[row * cols + k] = scale * sum;
    }
}

//input to this kerbel would be the output from the prev kernel (dct2d_rows_kernel)
__global__ void dct2d_cols_kernel(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < cols && k < rows) {
        float sum = 0.0f;
        float scale = (k == 0) ? sqrtf(1.0f / rows) : sqrtf(2.0f / rows);
        
        for (int n = 0; n < rows; n++) {
            sum += input[n * cols + col] * cosf(PI * k * (2 * n + 1) / (2.0f * rows));
        }
        
        output[k * cols + col] = scale * sum;
    }
}

void dct1d(const float* h_input, float* h_output, int N) {
    float *d_input, *d_output;
    
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    dct1d_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

void idct1d(const float* h_input, float* h_output, int N) {
    float *d_input, *d_output;
    
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    idct1d_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

void dct2d(const float* h_input, float* h_output, int rows, int cols) {
    float *d_input, *d_temp, *d_output;
    size_t size = rows * cols * sizeof(float);
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_temp, size);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + 15) / 16, (rows + 15) / 16); //hardcoded values for now ; will change later
    
    // Apply DCT to rows
    dct2d_rows_kernel<<<gridSize, blockSize>>>(d_input, d_temp, rows, cols);
    
    // Apply DCT to columns
    dct2d_cols_kernel<<<gridSize, blockSize>>>(d_temp, d_output, rows, cols);
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
}

// Example usage (for hardcoded values for now; gotta add code for image)
int main() {
    const int N = 8;
    float input[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float output[N];
    float reconstructed[N];
    
    printf("Original signal:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", input[i]);
    }
    printf("\n\n");
    
    dct1d(input, output, N);
    
    printf("DCT coefficients:\n");
    for (int i = 0; i < N; i++) {
        printf("%.4f ", output[i]);
    }
    printf("\n\n");
    
    idct1d(output, reconstructed, N);
    
    printf("Reconstructed signal:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", reconstructed[i]);
    }
    printf("\n\n");
    
    const int rows = 4, cols = 4;
    float input2d[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float output2d[16];
    
    printf("2D DCT input:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.0f ", input2d[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    dct2d(input2d, output2d, rows, cols);
    
    printf("2D DCT coefficients:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", output2d[i * cols + j]);
        }
        printf("\n");
    }
    
    return 0;
}