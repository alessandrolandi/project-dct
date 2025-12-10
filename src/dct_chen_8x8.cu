// naive implementation of chens algorithm on 1d and 2d dct for 8x8 matrices
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 8
#define BLOCK_SIZE 8 


__constant__ float C[7] = {
    0.980785f, // cos(pi/16)
    0.923880f, // cos(2*pi/16)
    0.831470f, // cos(3*pi/16)
    0.707107f, // cos(4*pi/16) = sqrt(2)/2
    0.555570f, // cos(5*pi/16)
    0.382683f, // cos(6*pi/16)
    0.195090f, // cos(7*pi/16
};

// 1/sqrt(N)
#define one_by_root_8 0.35355339f // 1 / sqrt(8)

// sqrt (2/8) = sqrt (1/4) = 1/2
#define half 0.5f 

// 1D 8x1 Fast DCT 

__device__ void dct_1d_8x1(float *data) {
    float x[N], y[N];

    for (int i = 0; i < N; i++) {
        x[i] = data[i];
    }

    // Butterfly    
    float t0 = x[0]+x[7];
    float t1 = x[1]+x[6];
    float t2 = x[2]+x[5];
    float t3 = x[3]+x[4];
    float t4 = x[3]-x[4];
    float t5 = x[2]-x[5];
    float t6 = x[1]-x[6];
    float t7 = x[0]-x[7];

    // Type 4 matrice (for even matrices)
    float b0 = t0+t3;
    float b1 = t1+t2;
    float b2 = t1-t2;
    float b3 = t0-t3;
    
    y[0] = (b0+b1) * one_by_root_8; 
    y[4] = (b0-b1) * half;

    y[2] = (b2*C[1] + b3*C[5]) * half;
    y[6] = (b3*C[1] - b2*C[5]) * half;

    float z1 = t4*C[6] + t7*C[0];
    float z2 = t5*C[2] + t6*C[4];
    float z3 = t6*C[2] - t5*C[4];
    float z4 = t7*C[6] - t4*C[0]; 

    y[1] = z1*half;
    y[3] = z2*half;
    y[5] = z3*half;
    y[7] = z4*half;

    for (int i = 0; i < N; i++) {
        data[i] = y[i];
    }
}

// 2D 8x8 DCT Global Kernel (using the seperable property) 

__global__ void dct_2d_8x8(const float *input_matrix, float *output_matrix, int width) {
    int block_x_offset = blockIdx.x * N;
    int block_y_offset = blockIdx.y * N;
    
    __shared__ float tile[N][N];
    
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    int global_row = block_y_offset + row;
    int global_col = block_x_offset + col;
    
    if (global_row < width && global_col < width) {
        tile[row][col] = input_matrix[global_row * width + global_col];
    }
    
    __syncthreads();
    
    if (row < N) {
        float row_data[N];
        for (int k = 0; k < N; k++) {
            row_data[k] = tile[row][k];
        }

        dct_1d_8x1(row_data);

        for (int k = 0; k < N; k++) {
            tile[row][k] = row_data[k];
        }
    }
    
    __syncthreads();
    
    if (col < N) {
        float col_data[N];
        for (int k = 0; k < N; k++) {
            col_data[k] = tile[k][col];
        }

        dct_1d_8x1(col_data);

        for (int k = 0; k < N; k++) {
            tile[k][col] = col_data[k];
        }
    }
    
    __syncthreads();
    
    if (global_row < width && global_col < width) {
        output_matrix[global_row * width + global_col] = tile[row][col];
    }
}


void print_matrix(const char *name, const float *matrix, int rows, int cols) {
    printf("\n--- %s ---\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}


void initialize_input(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (float)(i * cols + j + 1);
        }
    }
}


int main() {
    size_t size = N * N * sizeof(float);

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    initialize_input(h_input, N, N);
    print_matrix("Input 8x8 Matrix", h_input, N, N);

    float *d_input = NULL, *d_output = NULL;
    
    dim3 blockSize(N, N);
    dim3 gridSize(1,1); 

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    dct_2d_8x8<<<gridSize, blockSize>>>(d_input, d_output, N);
    
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, size, cudaMemcpyHostToDevice);

    print_matrix("Output 8x8 Matrix ", h_output, N, N);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
