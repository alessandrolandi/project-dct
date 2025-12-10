#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define PI 3.14159265358979323846f 


__device__ float c_k(int k) {
    return (k == 0) ? (1.0f / sqrtf(2.0f)) : 1.0f;
}

// 1D DCT-II kernel (Non-Orthogonal)
__global__ void dct1d_kernel(const float* input, float* output, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k < N) {
        float sum = 0.0f;
        
        for (int j = 0; j < N; j++) {
            sum += input[j] * cosf(PI * k * (2.0f * j + 1.0f) / (2.0f * N));
        }

        output[k] = (2.0f / N) * c_k(k) * sum;
    }
}

// 1D IDCT-II kernel 
__global__ void idct1d_kernel(const float* input, float* output, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < N) {
        float sum = 0.0f;
        
        for (int k = 0; k < N; k++) {
            sum += c_k(k) * input[k] * cosf(PI * k * (2.0f * j + 1.0f) / (2.0f * N));
        }
        
        output[j] = sum;
    }
}


__global__ void dct2d_rows_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int k_col = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (row < rows && k_col < cols) {
        float sum = 0.0f;
        
        for (int j_col = 0; j_col < cols; j_col++) { 
            sum += input[row * cols + j_col] * cosf(PI * k_col * (2.0f * j_col + 1.0f) / (2.0f * cols));
        }
        
        output[row * cols + k_col] = (2.0f / cols) * c_k(k_col) * sum;
    }
}


__global__ void dct2d_cols_kernel(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k_row = blockIdx.y * blockDim.y + threadIdx.y; 
    
    if (col < cols && k_row < rows) {
        float sum = 0.0f;
        
        for (int j_row = 0; j_row < rows; j_row++) { 
            sum += input[j_row * cols + col] * cosf(PI * k_row * (2.0f * j_row + 1.0f) / (2.0f * rows));
        }

        output[k_row * cols + col] = (2.0f / rows) * c_k(k_row) * sum;
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
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    
    dct2d_rows_kernel<<<gridSize, blockSize>>>(d_input, d_temp, rows, cols);
    dct2d_cols_kernel<<<gridSize, blockSize>>>(d_temp, d_output, rows, cols);
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
}


int main() {
    const int N = 4;
    float input[N] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[N];
    float reconstructed[N];
    
    printf("N=4 1D array original signal:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", input[i]);
    }
    printf("\n\n");
    
    dct1d(input, output, N);
    
    printf("DCT coefficients: \n");
    for (int i = 0; i < N; i++) {
        printf("%.4f ", output[i]);
    }
    printf("\n\n");
    
    idct1d(output, reconstructed, N);
    
    printf("Reconstructed signal: \n");
    for (int i = 0; i < N; i++) {
        printf("%.4f ", reconstructed[i]); 
    }
    printf("\n\n");

    // --- 1D 8x1 Example ---

    const int N8 = 8;
    float input8[N8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float output8[N8];
    float reconstructed8[N8];
    
    printf("N=8 1D array original signal:\n");
    for (int i = 0; i < N8; i++) {
        printf("%.2f ", input8[i]);
    }
    printf("\n\n");
    
    dct1d(input8, output8, N8);
    
    printf("DCT coefficients:\n");
    for (int i = 0; i < N8; i++) {
        printf("%.4f ", output8[i]);
    }
    printf("\n\n");
    
    idct1d(output8, reconstructed8, N8);
    
    printf("Reconstructed signal:\n");
    for (int i = 0; i < N8; i++) {
        printf("%.4f ", reconstructed8[i]); 
    }
    printf("\n\n");
    
    
    // --- 2D DCT Example ---
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
            printf("%7.2f ", input2d[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    dct2d(input2d, output2d, rows, cols);
    
    printf("2D DCT coefficients: \n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%7.2f ", output2d[i * cols + j]);
        }
        printf("\n");
    }
    
    return 0;
}
