#ifndef DCT_H
#define DCT_H

#ifdef __cplusplus
}
#endif

void dct2d(const float* h_input, float* h_output, int rows, int cols)
void idct_2d(float input[8][8], float output[8][8]);


#ifdef __cplusplus
}
#endif

#endif //DCT_H
