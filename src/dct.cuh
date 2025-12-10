#ifndef DCT_H
#define DCT_H

#ifdef __cplusplus
extern "C" {
#endif

// 1D DCT/IDCT
void dct1d(const float* h_input, float* h_output, int N);
void idct1d(const float* h_input, float* h_output, int N);

// 2D DCT/IDCT
void dct2d(const float* h_input, float* h_output, int rows, int cols);
void idct2d(const float* h_input, float* h_output, int rows, int cols);
void dct2d_fft(const float* h_input, float* h_output, int rows, int cols);

// CUDA kernels
__global__ void dct8x8_kernel(const float* input, float* output,
                              int width, int height);

__global__ void idct8x8_kernel(const float* input, float* output,
                               int width, int height);

#ifdef __cplusplus
}
#endif

#endif //DCT_H
