#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define PI 3.14159265358979323846


__host__ __device__ int bit_reverse(int x, int bits) {
    int result = 0;
    for (int i = 0; i < bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

__global__ void bit_reversal_kernel(int* d_input, int* d_output, int N, int bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        d_output[idx] = bit_reverse(d_input[idx], bits);
    }
}

// __device__ unsigned bit_reverse_fast(unsigned x, int bits) {
//     unsigned r = __brev(x);        
//     return r >> (32 - bits);       
// }

// ============ B_N and B_N* MATRICES ============

// B_N = [I_N/2, I_bar_N/2; I_bar_N/2, -I_N/2]
__global__ void apply_BN_matrix(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = N / 2;
    
    if (idx < half) {
        // First row block: [I_N/2, I_bar_N/2]
        output[idx] = input[idx] + input[N - 1 - idx];
        // printf("%0d %0f\n",idx,  output[idx]);
    } else if (idx < N ) {
        // Second row block: [I_bar_N/2, -I_N/2]
        int local_idx = idx - half;
        output[idx] = input[N - 1 - local_idx - half] - input[idx];
        // printf("%0d %0f\n",idx,  output[idx]);
    }
}

// B_N* = [-I_N/2, I_bar_N/2; I_bar_N/2, I_N/2]
__global__ void apply_BN_star_matrix(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = N / 2;
    
    if (idx < half) {
        // First row block: [-I_N/2, I_bar_N/2]
        output[idx] = -input[idx] + input[N - 1 - idx];
        // printf("%0d %0f\n",idx,  output[idx]);
    } else if (idx < N) {
        // Second row block: [I_bar_N/2, I_N/2]
        int local_idx = idx - half;
        output[idx] = input[N - 1 - local_idx - half] + input[idx];
        // printf("%0d %0f\n",idx,  output[idx]);
    }
}

// ============ TYPE 1 MATRIX (M1) ============

__global__ void apply_type1_matrix(const float* input, float* output, int N, int log2N) {

    // printf("[%0d, %0d]",N,log2N);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N) return;
    
    int half = N / 2;
    
    // Calculate j (row index within each half)
    int j = (idx < half) ? idx : (idx - half);
    // printf("[%0d]",j);
    
    // Bit-reverse: aj = bit_reverse(N/2 + j, bits-1)
    int aj = bit_reverse(half + j, log2N);
    // printf("[%0d]",aj);

    float angle = PI * aj / (2.0f * N);
    float s = sinf(angle);
    float c = cosf(angle);
    // printf("[%0f %0f]",s,c);
    
    // Anti-diagonal index: N-1-idx
    int anti_idx = N - 1 - idx;
    
    if (idx < half) {
        // Upper half: S on main diagonal, C on anti-diagonal
        output[idx] = s * input[idx] + c * input[anti_idx];
        // printf("[%0d %0f]",idx, output[idx]);
    } else {
        // Lower half: C on main diagonal, -S on anti-diagonal
        output[idx] = c * input[idx] - s * input[anti_idx];
        // printf("[%0d %0f]",idx, output[idx]);
    }
}

// ============ TYPE 2 MATRIX (Last Matrix) ============

__global__ void apply_type2_matrix(const float* input, float* output, int N) {


    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n8 = N / 8;
    float c = cosf(PI / 8.0f);
    
    if (idx < N) {
        if (idx < n8) {
            // First N/8: Identity
            output[idx] = input[idx];
        } 
        
        else if (idx < 2 * n8 && idx > n8) {
            // Next N/8: -C4^1 where C4^1 = cos(π/8)
            output[idx] = -c * input[idx] + c * input[idx+1];
        } 
        
        else if (idx < 3 * n8 && idx > 2 * n8 ) {
            // Next N/8: C4^1
            output[idx] = c * (input[idx]+input[idx+1]);
        } 
        
        else {
            // Last N/8: Identity
            output[idx] = input[idx];
        }
    }

    // printf("\n");
    // for (int i = 0; i < 4; i++) {
    //         printf("%.4f%s", output[idx], (i < N-1) ? ", " : "");
    // }
}

// ============ TYPE 4 MATRIX (Even matrices - Butterflies) ============

__global__ void apply_type4_matrix(const float* input, float* output, int N, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        int block_size = N / (1 << p);
        int block_idx = idx / block_size;
        int local_idx = idx % block_size;
        int half_block = block_size / 2;
        
        if (local_idx < half_block) {
            int pair_idx = idx + half_block;
            
            if (block_idx % 2 == 0) {
                // B_l butterfly
                output[idx] = input[idx] + input[pair_idx];
                output[pair_idx] = input[idx] - input[pair_idx];
            } else {
                // B_l* butterfly
                output[idx] = input[idx] - input[pair_idx];
                output[pair_idx] = input[idx] + input[pair_idx];
            }
        }
    }
}

// ============ DCT IMPLEMENTATION ============

// For N=4: A_4 = P_4 * [A_2, 0; 0, R_2] * B_4
// R_2 = M1 (Type 1 matrix) - no decomposition needed
// A_2 is the base case 2x2 DCT

__global__ void apply_A2_dct(const float* input, float* output) {
    // 2x2 DCT base case: (1/√2) * [[1,1],[1,-1]]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float scale = 1.0f / sqrtf(2.0f);
    
    if (idx == 0) {
        output[0] = scale * (input[0] + input[1]);
    } else if (idx == 1) {
        output[1] = scale * (input[0] - input[1]);
    }
}

// Apply block diagonal [A_N/2, 0; 0, R_N/2]
void apply_block_diagonal(const float* h_input, float* h_output, int N, int log2N) {
    int half = N / 2;

    // printf("[%0d]\n", N);
    
    float *d_input, *d_output, *d_temp;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_temp, N * sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 8;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Upper block: Apply A_N/2 recursively
    if (half == 2) {
        // Base case: 2x2 DCT
        apply_A2_dct<<<gridSize, blockSize>>>(d_input, d_output);
        // printf("\n");
        
    } else {
        // Recursive case
        float* h_upper = (float*)malloc(half * sizeof(float));
        float* h_upper_out = (float*)malloc(half * sizeof(float));
        cudaMemcpy(h_upper, d_input, half * sizeof(float), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < N; i++) {
        //     printf("%.4f%s", h_upper[i], (i < N-1) ? ", " : "");
        // }
        
        // Recursively apply A_N/2
        apply_block_diagonal(h_upper, h_upper_out, half, log2N);
        
        cudaMemcpy(d_output, h_upper_out, half * sizeof(float), cudaMemcpyHostToDevice);

        // printf("%0d\n", half);
        // for (int i = 0; i < 4; i++) {
        //     printf("%.4f%s", h_upper_out[i], (i < N-1) ? ", " : "");
        // }

        //  printf("\n");
        
        free(h_upper);
        free(h_upper_out);
    }
    
    // Lower block: Apply R_N/2 (decomposed into Type 1, 2, 3, 4 matrices)
    int num_R_matrices = 2 * log2N - 3;
    
    if (num_R_matrices == 1) {
        // N=4 case: R_2 is just Type 1 matrix
        float *d_lower_in, *d_lower_out;
        cudaMalloc(&d_lower_in, half * sizeof(float));
        cudaMalloc(&d_lower_out, half * sizeof(float));
        
        cudaMemcpy(d_lower_in, d_input + half, half * sizeof(float), cudaMemcpyDeviceToDevice);
        apply_type1_matrix<<<gridSize, blockSize>>>(d_lower_in, d_lower_out, half, log2N-1);
        cudaMemcpy(d_output + half, d_lower_out, half * sizeof(float), cudaMemcpyDeviceToDevice);
        
        cudaFree(d_lower_in);
        cudaFree(d_lower_out);
    } else {
        // N>=8: R_N/2 = M1 * M2 * M3 * ... * M(2log2N-3)
        float *d_lower_work1, *d_lower_work2;
        cudaMalloc(&d_lower_work1, half * sizeof(float));
        cudaMalloc(&d_lower_work2, half * sizeof(float));
        
        cudaMemcpy(d_lower_work1, d_input + half, half * sizeof(float), cudaMemcpyDeviceToDevice);
        
        float* current = d_lower_work1;
        float* next = d_lower_work2;
        
        //  if num_R_matrices=1 which is the case for N=4, only apply matrix type 1
        //  if num_R_matrices=3 which is the case for N=8, apply types 1, 2, 4

        for (int m = 1; m <= num_R_matrices; m++) {
            if (m == 1) {
                // Type 1: First matrix
                apply_type1_matrix<<<gridSize, blockSize>>>(current, next, half, log2N-1 );
            } 
            
            else if (m == num_R_matrices) {
                // Type 2: Last matrix
                apply_type2_matrix<<<gridSize, blockSize>>>(current, next, half);
            } 
            

            else {
                // Type 4: Even matrices
                apply_type4_matrix<<<gridSize, blockSize>>>(current, next, half, m / 2);
            }
            cudaDeviceSynchronize();
            
            // Swap buffers
            float* tmp = current;
            current = next;
            next = tmp;
        }
        
        cudaMemcpy(d_output + half, current, half * sizeof(float), cudaMemcpyDeviceToDevice);
        
        cudaFree(d_lower_work1);
        cudaFree(d_lower_work2);
    }
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < N; i++) {
    //         printf("%.4f%s", h_output[i], (i < N-1) ? ", " : "");
    // }

    // printf("\n");
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
}


// A_N = P_N * [A_N/2, 0; 0, R_N/2] * B_N
void fast_dct_chen(const float* h_input, float* h_output, int N) {
    int log2N = log2(N); // Get number of bits

    
    float *d_input, *d_bn_out, *d_block_out;
    
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_bn_out, N * sizeof(float));
    cudaMalloc(&d_block_out, N * sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 8;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Step 1 Apply B_N
    apply_BN_matrix<<<gridSize, blockSize>>>(d_input, d_bn_out, N);
    cudaDeviceSynchronize();
    
    float* h_bn_result = (float*)malloc(N * sizeof(float));
    cudaMemcpy(h_bn_result, d_bn_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Step 2 Apply [A_N/2, 0; 0, R_N/2]
    float* h_block_result = (float*)malloc(N * sizeof(float));
    apply_block_diagonal(h_bn_result, h_block_result, N, log2N);

    
    // Step 3 Apply P_N (bit-reversal permutation)
    for (int k = 0; k < N; k++) {
        int k_rev = bit_reverse(k, log2N);
        h_output[k] = h_block_result[k_rev];
    }
    
    // Step 4: Apply normalization
    for (int k = 0; k < N; k++) {
        h_output[k] = (2.0f / N )*h_output[k];
    }
    //
    free(h_bn_result);
    free(h_block_result);
    
    cudaFree(d_input);
    cudaFree(d_bn_out);
    cudaFree(d_block_out);
}

// ============ 2D DCT USING SEPARABLE PROPERTY ============

// Apply 1D DCT to rows, then to columns
void fast_dct_chen_2D(const float* h_input, float* h_output, int N) {
   
    printf("\n========================================\n");
    printf("        2D DCT: %dx%d matrix\n", N, N);
    printf("========================================\n\n");
    
    float* temp_result = (float*)malloc(N * N * sizeof(float));
    float* row_buffer = (float*)malloc(N * sizeof(float));
    float* col_buffer = (float*)malloc(N * sizeof(float));
    
    // Apply 1D DCT to each row;
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            row_buffer[c] = h_input[r * N + c];
        }
        
        float* row_out = (float*)malloc(N * sizeof(float));
        fast_dct_chen(row_buffer, row_out, N);
        
        for (int c = 0; c < N; c++) {
            temp_result[r * N + c] = row_out[c];
        }  
        free(row_out);
        
    }
    
    // Apply 1D DCT to each column
    for (int c = 0; c < N; c++) {
        for (int r = 0; r < N; r++) {
            col_buffer[r] = temp_result[r * N + c];
        }
        
        float* col_out = (float*)malloc(N * sizeof(float));
        fast_dct_chen(col_buffer, col_out, N);
        
        for (int r = 0; r < N; r++) {
            h_output[r + c * N] = col_out[r];
            
        }
        
        free(col_out);
    }
    
    free(temp_result);
    free(row_buffer);
    free(col_buffer);
}


void test_2d_dct(const float* input, int N) {
    printf("Input matrix:\n");
    for (int r = 0; r < N; r++) {
        printf("  ");
        for (int c = 0; c < N; c++) {
            printf("%5.1f ", input[r * N + c]);
        }
        printf("\n");
    }
    printf("\n");
    
    float* output = (float*)malloc(N * N * sizeof(float));
    fast_dct_chen_2D(input, output, N);
    
    printf("Output DCT coefficients:\n");
    for (int r = 0; r < N; r++) {
        printf("  ");
        for (int c = 0; c < N; c++) {
            printf("%8.2f ", output[r * N + c]);
        }
        printf("\n");
    }
    printf("\n");
    
    free(output);
}



int main() {
    printf("========================================\n");
    printf("        Chen's Fast DCT\n");
    printf("========================================\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Test N=4 (1D)
    const int N4 = 4;
    float input4[N4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output4[N4];
 
    

    cudaEventRecord(start, 0);
    fast_dct_chen(input4, output4, N4);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);


    printf("\n\n========================================\n");
    printf("--- Testing 1D DCT: N=4 ---\n");
    printf("========================================\n\n");

    printf("Input: [1, 2, 3, 4]\n");
    printf("Result:   [%.4f, %.4f, %.4f, %.4f]\n", 
           output4[0], output4[1], output4[2], output4[3]);
    printf("Expected: [3.5355 -1.5772 0.0000 -0.1121]\n");
    printf("\nTime Elapsed: %0f\n", elapsed_ms);
    
    // Test N=8 (1D)
    printf("\n\n========================================\n");
    printf("--- Testing 1D DCT: N=8 ---\n");
    printf("========================================\n\n");
    
    const int N8 = 8;
    float input8[N8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float output8[N8];
    
    printf("Input: [1, 2, 3, 4, 5, 6, 7, 8]\n");   
    cudaEventRecord(start, 0); 
    fast_dct_chen(input8, output8, N8);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("\nTime Elapsed: %0f\n", elapsed_ms);
    
    printf("Result: ");
    for (int i = 0; i < N8; i++) {
        printf("%.4f ", output8[i]);
    }
    printf("\nExpected: [6.3640 -3.2212 0.0000 -0.3367 0.0000 -0.1005 -0.0000 -0.0254 ]\n");
    
    printf("\n\n");
    
    // Test 2D DCT: 4x4
    printf("\n========================================\n");
    printf("--- Testing 2D DCT: 4x4 ---\n");
    printf("========================================\n");
    

    float input_4x4[16] = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16
    };
    
    cudaEventRecord(start, 0); 
    test_2d_dct(input_4x4, 4);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("\nTime Elapsed: %0f\n", elapsed_ms);
    
    // Test 2D DCT: 8x8
    printf("\n========================================\n");
    printf("--- Testing 2D DCT: 8x8 ---\n");
    printf("========================================\n");
    
    float input_8x8[64];
    for (int i = 0; i < 64; i++) {
        input_8x8[i] = (float)(1);
    }
    
    cudaEventRecord(start, 0);
    test_2d_dct(input_8x8, 8);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("\nTime Elapsed: %0f\n", elapsed_ms);

    return 0;
}
