#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "lib/ppm_parser.h"
#include "dct.h"


//JPEG quantization table
__constant__ int d_quant_table[64] = {
    16, 11, 10, 16, 24,  40,  51,  61,
    12, 12, 14, 19, 26,  58,  60,  55,
    14, 13, 16, 24, 40,  57,  69,  56,
    14, 17, 22, 29, 51,  87,  80,  62,
    18, 22, 37, 56, 68,  109, 103, 77,
    24, 35, 55, 64, 81,  104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99
};

//CUDA kernel: quantize DCT coefficients
__global__ void quantize_kernel(float *dct_blocks, int *quantized, int num_blocks, int quality) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (block_idx < num_blocks) {
        int scale = (quality < 50) ? (5000 / quality) : (200 - 2 * quality);
        
        for (int i = 0; i < 64; i++) {
            int q_value = (d_quant_table[i] * scale + 50) / 100;
            if (q_value < 1) q_value = 1;
            if (q_value > 255) q_value = 255;
            
            int idx = block_idx * 64 + i;
            quantized[idx] = (int)roundf(dct_blocks[idx] / q_value);
        }
    }
}

//CUDA kernel: dequantize DCT coefficients
__global__ void dequantize_kernel(int *quantized, float *dct_blocks, int num_blocks, int quality) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (block_idx < num_blocks) {
        int scale = (quality < 50) ? (5000 / quality) : (200 - 2 * quality);
        
        for (int i = 0; i < 64; i++) {
            int q_value = (d_quant_table[i] * scale + 50) / 100;
            if (q_value < 1) q_value = 1;
            if (q_value > 255) q_value = 255;
            
            int idx = block_idx * 64 + i;
            dct_blocks[idx] = quantized[idx] * q_value;
        }
    }
}

//compress image using DCT
PPMImage* compress_image(PPMImage *input, int quality) {
    if (!input || quality < 1 || quality > 100) {
        fprintf(stderr, "Error: Invalid input or quality\n");
        return NULL;
    }
    
    //create output image
    PPMImage *output = (PPMImage*)malloc(sizeof(PPMImage));
    output->width = input->width;
    output->height = input->height;
    output->max_val = input->max_val;
    output->pixels = (Pixel*)malloc(input->width * input->height * sizeof(Pixel));
    
    //calculate number of 8x8 blocks
    int blocks_x = (input->width + 7) / 8;
    int blocks_y = (input->height + 7) / 8;
    int num_blocks = blocks_x * blocks_y;
    
    
    //process each color channel
    for (int channel = 0; channel < 3; channel++) {
        const char *channel_name[] = {"Red", "Green", "Blue"};
        
        float *h_dct_blocks = (float*)malloc(num_blocks * 64 * sizeof(float));
        int *h_quantized = (int*)malloc(num_blocks * 64 * sizeof(int));
        
        //extract all blocks and apply DCT
        for (int by = 0; by < blocks_y; by++) {
            for (int bx = 0; bx < blocks_x; bx++) {
                float block[64];  
                float dct_block[64];
                
                //extract block as flat array
                for (int y = 0; y < 8; y++) {
                    for (int x = 0; x < 8; x++) {
                        int img_x = bx * 8 + x;
                        int img_y = by * 8 + y;
                        
                        if (img_x < input->width && img_y < input->height) {
                            int idx = img_y * input->width + img_x;
                            if (channel == 0) block[y * 8 + x] = input->pixels[idx].r;
                            else if (channel == 1) block[y * 8 + x] = input->pixels[idx].g;
                            else block[y * 8 + x] = input->pixels[idx].b;
                        } else {
                            block[y * 8 + x] = 0.0f;  //pad with zeros
                        }
                    }
                }
                
                //apply DCT (8 rows, 8 cols)
                dct2d(block, dct_block, 8, 8);
                
                //copy to flat array
                int block_idx = by * blocks_x + bx;
                for (int i = 0; i < 64; i++) {
                    h_dct_blocks[block_idx * 64 + i] = dct_block[i];
                }
            }
        }
        
        //allocate device memory
        float *d_dct_blocks;
        int *d_quantized;
        cudaMalloc(&d_dct_blocks, num_blocks * 64 * sizeof(float));
        cudaMalloc(&d_quantized, num_blocks * 64 * sizeof(int));
        
        //copy DCT blocks to device
        cudaMemcpy(d_dct_blocks, h_dct_blocks, 
                   num_blocks * 64 * sizeof(float), cudaMemcpyHostToDevice);
        
        //quantize on GPU
        int threads = 256;
        int blocks = (num_blocks + threads - 1) / threads;
        quantize_kernel<<<blocks, threads>>>(d_dct_blocks, d_quantized, num_blocks, quality);
        cudaDeviceSynchronize();
        
        //dequantize on GPU
        dequantize_kernel<<<blocks, threads>>>(d_quantized, d_dct_blocks, num_blocks, quality);
        cudaDeviceSynchronize();
        
        //copy back to host
        cudaMemcpy(h_dct_blocks, d_dct_blocks,
                   num_blocks * 64 * sizeof(float), cudaMemcpyDeviceToHost);
        
        //apply inverse DCT and insert blocks back
        for (int by = 0; by < blocks_y; by++) {
            for (int bx = 0; bx < blocks_x; bx++) {
                float dct_block[64];
                float reconstructed[64];
                
                //copy from flat array
                int block_idx = by * blocks_x + bx;
                for (int i = 0; i < 64; i++) {
                    dct_block[i] = h_dct_blocks[block_idx * 64 + i];
                }
                
                //apply inverse DCT (8 rows, 8 cols)
                idct2d(dct_block, reconstructed, 8, 8);
                
                //insert back into image
                for (int y = 0; y < 8; y++) {
                    for (int x = 0; x < 8; x++) {
                        int img_x = bx * 8 + x;
                        int img_y = by * 8 + y;
                        
                        if (img_x < output->width && img_y < output->height) {
                            int idx = img_y * output->width + img_x;
                            float val = reconstructed[y * 8 + x];
                            
                            //clamp to [-128, 127]
                            if (val < -128.0f) val = -128.0f;
                            if (val > 127.0f) val = 127.0f;
                            
                            if (channel == 0) output->pixels[idx].r = val;
                            else if (channel == 1) output->pixels[idx].g = val;
                            else output->pixels[idx].b = val;
                        }
                    }
                }
            }
        }
        
        //cleanup
        cudaFree(d_dct_blocks);
        cudaFree(d_quantized);
        free(h_dct_blocks);
        free(h_quantized);
    }
    
    return output;
}

//calculate peak signal to noise ratio between the original image and the compresseed image
double calculate_psnr(PPMImage *original, PPMImage *compressed) {
    if (!original || !compressed) return 0.0;
    
    double mse = 0.0;
    int num_pixels = original->width * original->height;
    
    for (int i = 0; i < num_pixels; i++) {
        //convert from [-128, 127] to [0, 255] for comparison
        double r_orig = original->pixels[i].r + 128.0;
        double g_orig = original->pixels[i].g + 128.0;
        double b_orig = original->pixels[i].b + 128.0;
        
        double r_comp = compressed->pixels[i].r + 128.0;
        double g_comp = compressed->pixels[i].g + 128.0;
        double b_comp = compressed->pixels[i].b + 128.0;
        
        double dr = r_orig - r_comp;
        double dg = g_orig - g_comp;
        double db = b_orig - b_comp;
        
        mse += dr * dr + dg * dg + db * db;
    }
    
    mse /= (num_pixels * 3.0);
    
    if (mse == 0.0) return INFINITY;
    
    return 10.0 * log10((255.0 * 255.0) / mse);
}


//usage: ./compress <input.ppm> <quality>
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input.ppm> <quality>\n", argv[0]);
        printf("  quality: 1-100 (higher = better quality, less compression)\n");
        return 1;
    }
    
    const char *input_file = argv[1];
    int quality = atoi(argv[2]);
    
    if (quality < 1 || quality > 100) {
        fprintf(stderr, "Quality must be between 1 and 100\n");
        return 1;
    }
    
    PPMImage *original = parse_ppm(input_file);
    if (!original) {
        fprintf(stderr, "Failed to load image\n");
        return 1;
    }
    
    printf("Image loaded: %dx%d, max_val=%d\n", 
           original->width, original->height, original->max_val);
    
    //compress image
    PPMImage *compressed = compress_image(original, quality);
    if (!compressed) {
        free_ppm(original);
        return 1;
    }
    
    //calculate PSNR
    double psnr = calculate_psnr(original, compressed);
    printf("\nPSNR: %.2f dB\n", psnr);
    
    //save compressed image
    char output_file[256];
    snprintf(output_file, sizeof(output_file), "compressed_q%d.ppm", quality);
    
    if (write_ppm(output_file, compressed, 1)) {
        printf("Successfully saved compressed image\n");
    } else {
        fprintf(stderr, "Failed to save image\n");
    }
    
    //cleanup
    free_ppm(original);
    free_ppm(compressed);
    
    return 0;
}