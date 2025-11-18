#ifndef PPM_PARSER_H
#define PPM_PARSER_H


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float r, g, b;
} Pixel;

typedef struct {
    int width;
    int height;
    int max_val;
    Pixel *pixels;  //flat array of pixels (row-major order)
} PPMImage;

PPMImage* parse_ppm(const char *filename);
void free_ppm(PPMImage *img);
int write_ppm(const char *filename, PPMImage *img, int binary);


#ifdef __cplusplus
}
#endif

#endif 