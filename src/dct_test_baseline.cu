#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define PI 3.1415


// Simple error check macro
#define CHECK_CUDA(call) do {                           \
    cudaError_t err = (call);                           \
    if (err != cudaSuccess) {                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n",       \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while(0)

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

int main() {
    // 1. Choose an image: assume we use a 512x512 grayscale raw image (e.g., Lena)
    const int rows = 512;
    const int cols = 512;
    const char* filename = "barbara_gray.raw";  // put the file in the same directory

    // 2. Read original 8-bit grayscale pixels
    size_t num_pixels = (size_t)rows * cols;
    unsigned char* img_u8 = (unsigned char*)malloc(num_pixels);
    if (!img_u8) {
        fprintf(stderr, "malloc img_u8 failed\n");
        return -1;
    }

    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "cannot open %s\n", filename);
        return -1;
    }
    size_t read_bytes = fread(img_u8, 1, num_pixels, fp);
    fclose(fp);
    if (read_bytes != num_pixels) {
        fprintf(stderr, "read %zu bytes, expected %zu\n", read_bytes, num_pixels);
        return -1;
    }

    // 3. Convert to float and do JPEG-style centering (subtract 128)
    float* img_f   = (float*)malloc(num_pixels * sizeof(float));
    float* dct_out = (float*)malloc(num_pixels * sizeof(float));
    if (!img_f || !dct_out) {
        fprintf(stderr, "malloc float buffers failed\n");
        return -1;
    }

    for (size_t i = 0; i < num_pixels; ++i) {
        img_f[i] = (float)img_u8[i] - 128.0f;   // [0,255] -> [-128,127]
    }

    // 4. Use CUDA events to time dct2d
    cudaEvent_t start, stop;
    CHECK_CUDA( cudaEventCreate(&start) );
    CHECK_CUDA( cudaEventCreate(&stop) );

    // Do a warmup run first (optional)
    dct2d(img_f, dct_out, rows, cols);
    CHECK_CUDA( cudaDeviceSynchronize() );

    // Actual timing: e.g., run 100 iterations and take the average
    int iters = 1;
    CHECK_CUDA( cudaEventRecord(start, 0) );
    for (int i = 0; i < iters; ++i) {
        dct2d(img_f, dct_out, rows, cols);
    }
    CHECK_CUDA( cudaEventRecord(stop, 0) );
    CHECK_CUDA( cudaEventSynchronize(stop) );

    float ms = 0.0f;
    CHECK_CUDA( cudaEventElapsedTime(&ms, start, stop) );

    double avg_ms = ms / iters;
    double pixels = (double)rows * (double)cols;
    double ns_per_pixel = (avg_ms * 1e6) / pixels;

    printf("Image size: %d x %d (%.0f pixels)\n", rows, cols, pixels);
    printf("Average DCT2D time: %.4f ms (over %d runs)\n", (float)avg_ms, iters);
    printf("Time per pixel: %.2f ns/pixel\n", (float)ns_per_pixel);

    // 5. Optional: print the first few DCT coefficients
    printf("\nFirst 8 DCT coeffs of first row:\n");
    for (int j = 0; j < 8; ++j) {
        printf("%.2f ", dct_out[j]);
    }
    printf("\n");

    // Cleanup
    CHECK_CUDA( cudaEventDestroy(start) );
    CHECK_CUDA( cudaEventDestroy(stop) );
    free(img_u8);
    free(img_f);
    free(dct_out);

    return 0;
}
