#ifndef PARSER_H
#define PARSER_H

#include <stdint.h>

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
    Pixel *pixels;  
} PPMImage;

//ppm functions
PPMImage* parse_ppm(const char *filename);
void free_ppm(PPMImage *img);
int write_ppm(const char *filename, PPMImage *img, int binary);


//JPEG encoding structures
typedef struct {
    uint16_t code[256];
    uint8_t size[256];
    uint8_t value[256];
    int num_symbols;
} HuffmanTable;

typedef struct {
    uint32_t bit_buffer;
    int bits_in_buffer;
    uint8_t *output_buffer;
    int buffer_size;
    int buffer_pos;
} BitstreamWriter;

//JPEG marker constants
#define JPEG_SOI   0xFFD8
#define JPEG_EOI   0xFFD9
#define JPEG_SOF0  0xFFC0
#define JPEG_DHT   0xFFC4
#define JPEG_DQT   0xFFDB
#define JPEG_SOS   0xFFDA
#define JPEG_APP0  0xFFE0

#ifdef __cplusplus
}
#endif

#endif 