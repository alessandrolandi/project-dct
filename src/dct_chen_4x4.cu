#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define PI 3.1415
#define Cos_pi_by_4 0.7071f 
#define Cos_pi_by_8 0.9238f
#define Sin_pi_by_8 0.3826f

// Chen's Algorithm
__global__ void dct1d_fast_4x1_kernel(const float* input, float* output, int N, bool is_row_transform) {
    int row_or_col_index = is_row_transform ? (blockIdx.y * blockDim.y + threadIdx.y) : (blockIdx.x * blockDim.x + threadIdx.x);

    if (row_or_col_index >= N) return;
    float f[4];
    
    if (is_row_transform) {
        for (int i = 0; i < 4; i++) {
            f[i] = input[row_or_col_index * N + i];
        }
    } else {

        for (int i = 0; i < 4; i++) {
            f[i] = input[i * N + row_or_col_index];
        }
    }


    float a0 = f[0]+f[3]; 
    float a1 = f[1]+f[2]; 
    float a2 = f[1]-f[2]; 
    float a3 = f[0]-f[3]; 

    float b0 = a0 + a1; 
    float b2 = a0 - a1; 


    float m1 = a3*Cos_pi_by_8;   
    float m2 = a2*Sin_pi_by_8;   
    float m3 = a3*Sin_pi_by_8;   
    float m4 = a2*Cos_pi_by_8;  
    

    float F0_unnorm = b0;
    float F2_unnorm = b2;
    float F1_unnorm = m1+m2;
    float F3_unnorm = m3-m4;
    
    float output_row[4];
    
    output_row[0] = F0_unnorm*0.5f;    
    output_row[2] = F2_unnorm*Cos_pi_by_4;          
    output_row[1] = F1_unnorm*Cos_pi_by_4;              
    output_row[3] = F3_unnorm*Cos_pi_by_4;   
    float C_3pi_8 = 0.38268f; 
    float C_pi_8 = 0.92388f;  
    float R2_in1 = a3;
    float R2_in2 = a2;
    
    float F1_out = R2_in1 * C_pi_8 + R2_in2 * C_3pi_8; 
    float F3_out = R2_in1 * C_3pi_8 - R2_in2 * C_pi_8; 
    float F0_bit = b0;
    float F2_bit = b2;
    float F1_bit = F1_out;
    float F3_bit = F3_out;

    output_row[0] = F0_bit*0.5f;      
    output_row[2] = F2_bit*Cos_pi_by_4;       
    output_row[1] = F1_bit;              
    output_row[3] = F3_bit;


    if (is_row_transform) {
        for (int i = 0; i < 4; i++) {
            output[row_or_col_index * N + i]=output_row[i];
        }
    } else {
        for (int i = 0; i < 4; i++) {
            output[i * N + row_or_col_index]=output_row[i];
        }
    }
}


__host__ void dct2d(const float* h_input, float* h_output, int N) {
    float *d_input, *d_temp, *d_output;
    size_t size = N * N * sizeof(float);
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_temp, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 blockSize(4, 4); 
    dim3 gridSize(1, 1); 

    dct1d_fast_4x1_kernel<<<gridSize, blockSize>>>(d_input, d_temp, N, true); // row-wise
    dct1d_fast_4x1_kernel<<<gridSize, blockSize>>>(d_temp, d_output, N, false); // column-wise
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
}

__host__ void dct1d(const float* h_input, float* h_output, int N) {
    float *d_input, *d_output;
    size_t size = N * sizeof(float);
   
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 blockSize(1, 1); 
    dim3 gridSize(1, 1); 

    dct1d_fast_4x1_kernel<<<gridSize, blockSize>>>(d_input, d_output, N, true);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}



int main() {
    const int N = 4;
    float input_1d[N] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output_1d[N];

    printf("=====================================================\n");
    printf("DCT optimized with Chen's Algorithm, N=4 version\n");
    printf("=====================================================\n");

    printf("1D Input. \n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", input_1d[i]);
    }
    printf("\n\n");

    dct1d(input_1d, output_1d, N);
    
    printf("FDCT coefficients:\n");
    for (int i = 0; i < N; i++) {
        printf("%.4f ", output_1d[i]);
    }
    printf("\n\n");
    

    float input2d[N*N] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float output2d[N*N];
    
    printf("2D Input:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.0f ", input2d[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    dct2d(input2d, output2d, N);
    
    printf("2D FDCT coefficients:\n");
    for (int i = 0; i < N; i++) { 
        for (int j = 0; j < N; j++) {
            printf("%.2f ", output2d[i * N + j]);
        }
        printf("\n");
    }
    
    return 0;
}
