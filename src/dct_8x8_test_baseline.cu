#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define PI 3.1415926535f

// Simple error check macro
#define CHECK_CUDA(call) do {                           \
    cudaError_t err = (call);                           \
    if (err != cudaSuccess) {                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n",       \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while(0)

// ------------------------
// Block based DCT kernel for an eight by eight tile
// Each CUDA block processes one tile
// Each thread computes one DCT coefficient in the tile
// ------------------------
__global__ void dct8x8_kernel(const float* input, float* output,
                              int width, int height)
{
    // Tile origin in the image for this CUDA block
    int block_x = blockIdx.x;        
    int block_y = blockIdx.y;        

    int x0 = block_x * 8;            
    int y0 = block_y * 8;            

    // Thread position inside the tile, used as frequency indices v and u
    int v = threadIdx.x;             
    int u = threadIdx.y;             

    // Bounds check for image borders
    if (x0 + v >= width || y0 + u >= height) return;

    // Scaling factors alpha for DCT
    float alpha_u = (u == 0) ? sqrtf(1.0f / 8.0f) : sqrtf(2.0f / 8.0f);
    float alpha_v = (v == 0) ? sqrtf(1.0f / 8.0f) : sqrtf(2.0f / 8.0f);

    float sum = 0.0f;

    // Direct implementation of the two dimensional DCT formula using cosine
    // Here x and y are coordinates inside the tile
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            // Fetch the pixel inside the tile at position x and y
            int img_x = x0 + x;
            int img_y = y0 + y;
            float pixel = input[img_y * width + img_x];

            float cv = cosf( PI * (2.0f * x + 1.0f) * v / 16.0f );
            float cu = cosf( PI * (2.0f * y + 1.0f) * u / 16.0f );

            sum += pixel * cu * cv;
        }
    }

    float coeff = alpha_u * alpha_v * sum;

    // Store the coefficient F at position u and v back to the output image
    // Coefficients are written into the same spatial region as the input tile
    int out_x = x0 + v;
    int out_y = y0 + u;
    output[out_y * width + out_x] = coeff;
}

// ------------------------
// Block based inverse DCT kernel for an eight by eight tile
// Each thread computes one reconstructed pixel inside the tile
// ------------------------
__global__ void idct8x8_kernel(const float* input, float* output,
                               int width, int height)
{
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    int x0 = block_x * 8;
    int y0 = block_y * 8;

    // Spatial coordinates inside the tile
    int x = threadIdx.x;     
    int y = threadIdx.y;     

    if (x0 + x >= width || y0 + y >= height) return;

    float sum = 0.0f;

    // Direct implementation of the inverse DCT formula
    // Reconstruct f at position x and y from all frequency coefficients
    for (int u = 0; u < 8; ++u) {
        for (int v = 0; v < 8; ++v) {
            int coeff_x = x0 + v;
            int coeff_y = y0 + u;
            float Cuv = input[coeff_y * width + coeff_x];

            float alpha_u = (u == 0) ? sqrtf(1.0f / 8.0f) : sqrtf(2.0f / 8.0f);
            float alpha_v = (v == 0) ? sqrtf(1.0f / 8.0f) : sqrtf(2.0f / 8.0f);

            float cv = cosf( PI * (2.0f * x + 1.0f) * v / 16.0f );
            float cu = cosf( PI * (2.0f * y + 1.0f) * u / 16.0f );

            sum += alpha_u * alpha_v * Cuv * cu * cv;
        }
    }

    // Store the reconstructed pixel value, still in the centered range
    int out_x = x0 + x;
    int out_y = y0 + y;
    output[out_y * width + out_x] = sum;
}

int main()
{
    // Image size is rows by cols and both are divisible by eight
    const int rows = 512;
    const int cols = 512;
    const char* filename = "barbara_gray.raw";  

    size_t num_pixels = (size_t)rows * cols;

    // Load original eight bit grayscale pixels
    unsigned char* img_u8 = (unsigned char*)malloc(num_pixels);
    if (!img_u8) {
        fprintf(stderr, "malloc img_u8 failed\n");
        return -1;
    }

    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "cannot open %s\n", filename);
        free(img_u8);
        return -1;
    }
    size_t read_bytes = fread(img_u8, 1, num_pixels, fp);
    fclose(fp);
    if (read_bytes != num_pixels) {
        fprintf(stderr, "read %zu bytes, expected %zu\n", read_bytes, num_pixels);
        free(img_u8);
        return -1;
    }

    // Convert pixels to float and center values as in JPEG by subtracting a constant
    float* img_f      = (float*)malloc(num_pixels * sizeof(float));
    float* dct_blocks = (float*)malloc(num_pixels * sizeof(float));
    float* recon_f    = (float*)malloc(num_pixels * sizeof(float));  

    if (!img_f || !dct_blocks || !recon_f) {
        fprintf(stderr, "malloc float buffers failed\n");
        free(img_u8);
        return -1;
    }

    for (size_t i = 0; i < num_pixels; ++i) {
        img_f[i] = (float)img_u8[i] - 128.0f;   
    }

    // Allocate GPU memory
    float *d_input = nullptr, *d_dct = nullptr, *d_recon = nullptr;
    CHECK_CUDA( cudaMalloc(&d_input, num_pixels * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&d_dct,   num_pixels * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&d_recon, num_pixels * sizeof(float)) );

    // Copy input image to GPU
    CHECK_CUDA( cudaMemcpy(d_input, img_f,
                           num_pixels * sizeof(float),
                           cudaMemcpyHostToDevice) );

    // Configure grid and block for JPEG style block DCT
    dim3 blockSize(8, 8);                        
    dim3 gridSize(cols / 8, rows / 8);           

    // Run the block based DCT once and measure time
    cudaEvent_t start, stop;
    CHECK_CUDA( cudaEventCreate(&start) );
    CHECK_CUDA( cudaEventCreate(&stop) );

    CHECK_CUDA( cudaEventRecord(start, 0) );
    dct8x8_kernel<<<gridSize, blockSize>>>(d_input, d_dct, cols, rows);
    CHECK_CUDA( cudaEventRecord(stop, 0) );
    CHECK_CUDA( cudaEventSynchronize(stop) );

    float ms = 0.0f;
    CHECK_CUDA( cudaEventElapsedTime(&ms, start, stop) );

    double pixels = (double)rows * (double)cols;
    double ns_per_pixel = (ms * 1e6) / pixels;

    printf("Image size: %d x %d (%.0f pixels)\n", rows, cols, pixels);
    printf("8x8 block-based DCT time: %.4f ms\n", ms);
    printf("Time per pixel (DCT only): %.2f ns/pixel\n", (float)ns_per_pixel);

    // Copy DCT coefficients back to host memory
    CHECK_CUDA( cudaMemcpy(dct_blocks, d_dct,
                           num_pixels * sizeof(float),
                           cudaMemcpyDeviceToHost) );

    printf("\nFirst 8 DCT coeffs of the first 8x8 block (u=0,v=0..7, y0=0):\n");
    for (int v = 0; v < 8; ++v) {
        int idx = 0 * cols + v;  
        printf("%.2f ", dct_blocks[idx]);
    }
    printf("\n");

    // Run block based inverse DCT to reconstruct the image
    dct8x8_kernel<<<gridSize, blockSize>>>(d_input, d_dct, cols, rows); 
    idct8x8_kernel<<<gridSize, blockSize>>>(d_dct, d_recon, cols, rows);

    CHECK_CUDA( cudaMemcpy(recon_f, d_recon,
                           num_pixels * sizeof(float),
                           cudaMemcpyDeviceToHost) );

    // Add the centering constant back and write a raw file for visual inspection
    FILE* fp_out = fopen("recon_gray.raw", "wb");
    if (fp_out) {
        for (size_t i = 0; i < num_pixels; ++i) {
            float val = recon_f[i] + 128.0f;    
            if (val < 0.0f)   val = 0.0f;
            if (val > 255.0f) val = 255.0f;
            unsigned char out = (unsigned char)(val + 0.5f);
            fwrite(&out, 1, 1, fp_out);
        }
        fclose(fp_out);
        printf("Reconstructed image written to recon_gray.raw\n");
    } else {
        printf("Cannot open recon_gray.raw for writing\n");
    }

    // Release resources
    CHECK_CUDA( cudaEventDestroy(start) );
    CHECK_CUDA( cudaEventDestroy(stop) );
    CHECK_CUDA( cudaFree(d_input) );
    CHECK_CUDA( cudaFree(d_dct) );
    CHECK_CUDA( cudaFree(d_recon) );

    free(img_u8);
    free(img_f);
    free(dct_blocks);
    free(recon_f);

    return 0;
}
