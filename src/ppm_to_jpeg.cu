#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "lib/parser.h"
#include "dct.cuh"

//standard JPEG quantization tables
__constant__ uint8_t d_quant_luma[64] = {
    16, 11, 10, 16,  24,  40,  51,  61,
    12, 12, 14, 19,  26,  58,  60,  55,
    14, 13, 16, 24,  40,  57,  69,  56,
    14, 17, 22, 29,  51,  87,  80,  62,
    18, 22, 37, 56,  68, 109, 103,  77,
    24, 35, 55, 64,  81, 104, 113,  92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103,  99
};

__constant__ uint8_t d_quant_chroma[64] = {
    17, 18, 18, 24, 30,  40,  51,  61,
    18, 21, 24, 30, 40,  58,  60,  55,
    18, 24, 26, 35, 50,  57,  69,  56,
    24, 30, 35, 40, 60,  80,  80,  70,
    30, 40, 50, 60, 70,  95,  95,  80,
    40, 58, 57, 69, 80,  95, 105,  90,
    51, 60, 69, 80, 95, 110, 115, 100,
    61, 55, 56, 70, 80,  90, 100,  95
};

//standard Huffman tables
static const uint8_t std_dc_luminance_bits[16] = {
    0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0
};
static const uint8_t std_dc_luminance_vals[12] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
};

static const uint8_t std_ac_luminance_bits[16] = {
    0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125
};
static const uint8_t std_ac_luminance_vals[162] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
    0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
    0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
    0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
};

//zigzag order for 8x8 blocks 
static const uint8_t zigzag[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

//cuda kernel: quantize DCT coefficients
__global__ void quantize_kernel(float *dct_blocks, int16_t *quantized,
                                int num_blocks, int blocks_per_channel, int quality) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_idx < num_blocks) {
        //quality scaling
        float scale;
        if (quality >= 98) {
            scale = (100.0f - quality) * 0.05f;
            if (scale < 0.001f) scale = 0.001f;
        } else if (quality >= 90) {
            scale = (100.0f - quality) / 50.0f;
        } else if (quality >= 50) {
            scale = (100.0f - quality) / 10.0f;
        } else {
            scale = 5000.0f / quality / 100.0f;
        }

        //determine which channel
        int channel = block_idx / blocks_per_channel;
        const uint8_t *quant_table = (channel == 0) ? d_quant_luma : d_quant_chroma;

        for (int i = 0; i < 64; i++) {
            float q_value = quant_table[i] * scale;
            if (q_value < 1.0f) q_value = 1.0f;

            int idx = block_idx * 64 + i;
            quantized[idx] = (int16_t)roundf(dct_blocks[idx] / q_value);
        }
    }
}

//build Huffman lookup table
static void build_huffman_table(HuffmanTable *ht, const uint8_t *bits, const uint8_t *vals) {
    int code = 0;
    int si = 0;

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < bits[i]; j++) {
            ht->size[si] = i + 1;
            ht->value[si] = vals[si];
            ht->code[si] = code;
            si++;
            code++;
        }
        code <<= 1;
    }
    ht->num_symbols = si;
}

//initialize bitstream writer
static void init_bitstream(BitstreamWriter *bs) {
    bs->bit_buffer = 0;
    bs->bits_in_buffer = 0;
    bs->buffer_size = 1024 * 1024;  //1MB initial
    bs->output_buffer = (uint8_t*)malloc(bs->buffer_size);
    bs->buffer_pos = 0;
}

//write bits to buffer
static void put_bits(BitstreamWriter *bs, uint32_t bits, int nbits) {
    bs->bit_buffer = (bs->bit_buffer << nbits) | (bits & ((1 << nbits) - 1));
    bs->bits_in_buffer += nbits;

    while (bs->bits_in_buffer >= 8) {
        bs->bits_in_buffer -= 8;
        uint8_t byte = (bs->bit_buffer >> bs->bits_in_buffer) & 0xFF;

        //check buffer size
        if (bs->buffer_pos >= bs->buffer_size - 2) {
            bs->buffer_size *= 2;
            bs->output_buffer = (uint8_t*)realloc(bs->output_buffer, bs->buffer_size);
        }

        bs->output_buffer[bs->buffer_pos++] = byte;

        //byte stuffing: if 0xFF, write 0x00 after it
        if (byte == 0xFF) {
            bs->output_buffer[bs->buffer_pos++] = 0x00;
        }
    }
}

//finish writing bits (pad with 1s)
static void finish_bits(BitstreamWriter *bs) {
    if (bs->bits_in_buffer > 0) {
        bs->bit_buffer <<= (8 - bs->bits_in_buffer);
        bs->bit_buffer |= (1 << (8 - bs->bits_in_buffer)) - 1;
        bs->bits_in_buffer = 8;
        put_bits(bs, 0, 0);  //flush
    }
}

//compute category for coefficient
static void compute_category(int val, int *category, int *bits_val) {
    int abs_val = (val < 0) ? -val : val;
    *category = 0;

    int temp = abs_val;
    while (temp > 0) {
        (*category)++;
        temp >>= 1;
    }

    if (*category == 0) {
        *bits_val = 0;
    } else {
        if (val > 0) {
            *bits_val = val;
        } else {
            *bits_val = val - 1;
        }
    }
}

//encode one Huffman symbol
static void encode_huffman(BitstreamWriter *bs, HuffmanTable *ht, int symbol) {
    for (int i = 0; i < ht->num_symbols; i++) {
        if (ht->value[i] == symbol) {
            put_bits(bs, ht->code[i], ht->size[i]);
            return;
        }
    }
    fprintf(stderr, "Warning: Huffman symbol %d not found\n", symbol);
}

//encode one 8x8 block
static void encode_block(BitstreamWriter *bs, HuffmanTable *dc_table,
                        HuffmanTable *ac_table, int16_t *block, int *dc_pred) {
    //encode DC coefficient (differential)
    int dc_diff = block[0] - *dc_pred;
    *dc_pred = block[0];

    int category, bits_val;
    compute_category(dc_diff, &category, &bits_val);
    encode_huffman(bs, dc_table, category);
    if (category > 0) {
        put_bits(bs, bits_val & ((1 << category) - 1), category);
    }

    //encode AC coefficients in zigzag order
    int run_length = 0;
    int last_nonzero = 0;

    //find last non-zero coefficient
    for (int k = 63; k >= 1; k--) {
        if (block[zigzag[k]] != 0) {
            last_nonzero = k;
            break;
        }
    }

    for (int k = 1; k <= last_nonzero; k++) {
        int coeff = block[zigzag[k]];

        if (coeff == 0) {
            run_length++;

            //zrl: run length exceeded 15
            if (run_length == 16) {
                encode_huffman(bs, ac_table, 0xF0);
                run_length = 0;
            }
        } else {
            //encode any pending zero runs
            while (run_length >= 16) {
                encode_huffman(bs, ac_table, 0xF0);
                run_length -= 16;
            }

            compute_category(coeff, &category, &bits_val);
            int symbol = (run_length << 4) | category;
            encode_huffman(bs, ac_table, symbol);
            put_bits(bs, bits_val & ((1 << category) - 1), category);
            run_length = 0;
        }
    }

    //EOB: end of block
    if (last_nonzero < 63) {
        encode_huffman(bs, ac_table, 0x00);
    }
}

//write marker
static void write_marker(FILE *fp, uint16_t marker) {
    uint8_t bytes[2] = {(marker >> 8) & 0xFF, marker & 0xFF};
    fwrite(bytes, 1, 2, fp);
}

//write 16-bit big-endian
static void write_u16(FILE *fp, uint16_t val) {
    uint8_t bytes[2] = {(val >> 8) & 0xFF, val & 0xFF};
    fwrite(bytes, 1, 2, fp);
}

//write APP0 (JFIF) segment
static void write_app0(FILE *fp) {
    write_marker(fp, JPEG_APP0);
    write_u16(fp, 16);
    fwrite("JFIF\0", 1, 5, fp);
    fwrite("\x01\x01", 1, 2, fp);  //Version 1.1
    fwrite("\x00", 1, 1, fp);      //Units
    write_u16(fp, 1);              //X density
    write_u16(fp, 1);              //Y density
    fwrite("\x00\x00", 1, 2, fp);  //Thumbnail
}

//write DQT segment
static void write_dqt(FILE *fp, int quality) {
    //quality scaling
    float scale;
    if (quality >= 98) {
        scale = (100.0f - quality) * 0.05f;
        if (scale < 0.001f) scale = 0.001f;
    } else if (quality >= 90) {
        scale = (100.0f - quality) / 50.0f;
    } else if (quality >= 50) {
        scale = (100.0f - quality) / 10.0f;
    } else {
        scale = 5000.0f / quality / 100.0f;
    }

    //luminance quantization table
    uint8_t std_quant_luma[64] = {
        16, 11, 10, 16,  24,  40,  51,  61,
        12, 12, 14, 19,  26,  58,  60,  55,
        14, 13, 16, 24,  40,  57,  69,  56,
        14, 17, 22, 29,  51,  87,  80,  62,
        18, 22, 37, 56,  68, 109, 103,  77,
        24, 35, 55, 64,  81, 104, 113,  92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103,  99
    };

    write_marker(fp, JPEG_DQT);
    write_u16(fp, 67);
    uint8_t table_id = 0;
    fwrite(&table_id, 1, 1, fp);

    uint8_t quant_table[64];
    for (int i = 0; i < 64; i++) {
        int q = (int)(std_quant_luma[i] * scale + 0.5f);
        if (q < 1) q = 1;
        if (q > 255) q = 255;
        quant_table[i] = q;
    }

    fwrite(quant_table, 1, 64, fp);

    //chrominance quantization table
    uint8_t std_quant_chroma[64] = {
        17, 18, 18, 24, 30,  40,  51,  61,
        18, 21, 24, 30, 40,  58,  60,  55,
        18, 24, 26, 35, 50,  57,  69,  56,
        24, 30, 35, 40, 60,  80,  80,  70,
        30, 40, 50, 60, 70,  95,  95,  80,
        40, 58, 57, 69, 80,  95, 105,  90,
        51, 60, 69, 80, 95, 110, 115, 100,
        61, 55, 56, 70, 80,  90, 100,  95
    };

    write_marker(fp, JPEG_DQT);
    write_u16(fp, 67);
    table_id = 1;
    fwrite(&table_id, 1, 1, fp);

    for (int i = 0; i < 64; i++) {
        int q = (int)(std_quant_chroma[i] * scale + 0.5f);
        if (q < 1) q = 1;
        if (q > 255) q = 255;
        quant_table[i] = q;
    }

    fwrite(quant_table, 1, 64, fp);
}

//write SOF0 segment
static void write_sof0(FILE *fp, int width, int height) {
    write_marker(fp, JPEG_SOF0);
    write_u16(fp, 17);  //length for 3 components

    uint8_t precision = 8;
    fwrite(&precision, 1, 1, fp);
    write_u16(fp, height);
    write_u16(fp, width);

    uint8_t num_components = 3;  //rgb
    fwrite(&num_components, 1, 1, fp);

    //write Y, Cb, Cr components
    for (int i = 0; i < 3; i++) {
        uint8_t comp_id = i + 1;       //1=Y, 2=Cb, 3=Cr
        uint8_t sampling = 0x11;       //1x1 sampling (no subsampling)
        uint8_t qt_id = (i == 0) ? 0 : 1;  //Y uses table 0, Cb/Cr use table 1
        fwrite(&comp_id, 1, 1, fp);
        fwrite(&sampling, 1, 1, fp);
        fwrite(&qt_id, 1, 1, fp);
    }
}

//write DHT segment
static void write_dht(FILE *fp, uint8_t table_class, uint8_t table_id,
                      const uint8_t *bits, const uint8_t *vals, int num_vals) {
    write_marker(fp, JPEG_DHT);
    write_u16(fp, 19 + num_vals);

    uint8_t tc_th = (table_class << 4) | table_id;
    fwrite(&tc_th, 1, 1, fp);
    fwrite(bits, 1, 16, fp);
    fwrite(vals, 1, num_vals, fp);
}

//write SOS segment
static void write_sos(FILE *fp) {
    write_marker(fp, JPEG_SOS);
    write_u16(fp, 12);  //length for 3 components

    uint8_t num_components = 3;  //rgb
    fwrite(&num_components, 1, 1, fp);

    //write component selectors for rgb
    for (int i = 0; i < 3; i++) {
        uint8_t comp_id = i + 1;      
        uint8_t tables = 0x00;         //DC0, AC0 
        fwrite(&comp_id, 1, 1, fp);
        fwrite(&tables, 1, 1, fp);
    }

    uint8_t spectral[3] = {0, 63, 0};  //start, end, successive approximation
    fwrite(spectral, 1, 3, fp);
}

//main compression function
int compress_jpeg(const char *input_file, const char *output_file, int quality) {
    //load PPM image
    PPMImage *ppm_img = parse_ppm(input_file);

    if (!ppm_img) {
        fprintf(stderr, "Error: Could not load PPM image: %s\n", input_file);
        fprintf(stderr, "This program only accepts PPM input files\n");
        return 0;
    }

    int width = ppm_img->width;
    int height = ppm_img->height;
    Pixel *pixels = ppm_img->pixels;

    //calculate blocks
    int blocks_x = (width + 7) / 8;
    int blocks_y = (height + 7) / 8;
    int num_blocks = blocks_x * blocks_y;

    //allocate for 3 color channels rgb
    float *h_dct_blocks = (float*)malloc(3 * num_blocks * 64 * sizeof(float));
    int16_t *h_quantized = (int16_t*)malloc(3 * num_blocks * 64 * sizeof(int16_t));


    //process each color channel convert rgb to YCbCr
    for (int channel = 0; channel < 3; channel++) {
        const char *channel_name[] = {"Y (Luminance)", "Cb (Blue Chroma)", "Cr (Red Chroma)"};

        for (int by = 0; by < blocks_y; by++) {
            for (int bx = 0; bx < blocks_x; bx++) {
                float block[64];
                float dct_block[64];

                //extract 8x8 block and convert rgb to YCbCr
                for (int y = 0; y < 8; y++) {
                    for (int x = 0; x < 8; x++) {
                        int img_x = bx * 8 + x;
                        int img_y = by * 8 + y;

                        if (img_x < width && img_y < height) {
                            int idx = img_y * width + img_x;

                            float r = pixels[idx].r;
                            float g = pixels[idx].g;
                            float b = pixels[idx].b;

                            //convert rgb to YCbCr 
                            //Y  =  0.299*R + 0.587*G + 0.114*B
                            //Cb = -0.168736*R - 0.331264*G + 0.5*B
                            //Cr =  0.5*R - 0.418688*G - 0.081312*B
                            if (channel == 0) {
                                //Y (luminance)
                                block[y * 8 + x] = 0.299f * r + 0.587f * g + 0.114f * b;
                            } else if (channel == 1) {
                                //Cb (blue chroma)
                                block[y * 8 + x] = -0.168736f * r - 0.331264f * g + 0.5f * b;
                            } else {
                                //Cr (red chroma)
                                block[y * 8 + x] = 0.5f * r - 0.418688f * g - 0.081312f * b;
                            }
                        } else {
                            block[y * 8 + x] = 0.0f;
                        }
                    }
                }

                //apply 2D DCT using reference implementation
                dct2d(block, dct_block, 8, 8);

                //copy to flat array (channel offset + block offset)
                int block_idx = channel * num_blocks + by * blocks_x + bx;
                memcpy(&h_dct_blocks[block_idx * 64], dct_block, 64 * sizeof(float));
            }
        }
    }

    //allocate device memory for 3 channels
    float *d_dct_blocks;
    int16_t *d_quantized;
    int total_blocks = 3 * num_blocks;  //3 color channels
    cudaMalloc(&d_dct_blocks, total_blocks * 64 * sizeof(float));
    cudaMalloc(&d_quantized, total_blocks * 64 * sizeof(int16_t));

    //copy to device
    cudaMemcpy(d_dct_blocks, h_dct_blocks, total_blocks * 64 * sizeof(float), cudaMemcpyHostToDevice);

    //quantize on GPU (all 3 channels)
    int threads = 256;
    int blocks = (total_blocks + threads - 1) / threads;
    quantize_kernel<<<blocks, threads>>>(d_dct_blocks, d_quantized, total_blocks, num_blocks, quality);
    cudaDeviceSynchronize();

    //copy back quantized coefficients
    cudaMemcpy(h_quantized, d_quantized, total_blocks * 64 * sizeof(int16_t), cudaMemcpyDeviceToHost);

    //build Huffman tables
    HuffmanTable dc_table, ac_table;
    build_huffman_table(&dc_table, std_dc_luminance_bits, std_dc_luminance_vals);
    build_huffman_table(&ac_table, std_ac_luminance_bits, std_ac_luminance_vals);

    //initialize bitstream
    BitstreamWriter bs;
    init_bitstream(&bs);

    //encode all 3 channels 
    int dc_pred[3] = {0, 0, 0};  //separate DC predictor for each channel

    for (int by = 0; by < blocks_y; by++) {
        for (int bx = 0; bx < blocks_x; bx++) {
            int block_pos = by * blocks_x + bx;

            //encode rgb for this block position
            for (int channel = 0; channel < 3; channel++) {
                int block_idx = channel * num_blocks + block_pos;
                encode_block(&bs, &dc_table, &ac_table,
                           &h_quantized[block_idx * 64], &dc_pred[channel]);
            }
        }
    }
    finish_bits(&bs);

    //write JPEG file
    FILE *fp = fopen(output_file, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open output file\n");
        return 0;
    }

    //write headers
    write_marker(fp, JPEG_SOI);
    write_app0(fp);
    write_dqt(fp, quality);
    write_sof0(fp, width, height);
    write_dht(fp, 0, 0, std_dc_luminance_bits, std_dc_luminance_vals, 12);
    write_dht(fp, 1, 0, std_ac_luminance_bits, std_ac_luminance_vals, 162);
    write_sos(fp);

    //write entropy-coded data
    fwrite(bs.output_buffer, 1, bs.buffer_pos, fp);

    //write EOI
    write_marker(fp, JPEG_EOI);

    fclose(fp);

    //cleanup
    cudaFree(d_dct_blocks);
    cudaFree(d_quantized);
    free(h_dct_blocks);
    free(h_quantized);
    free(bs.output_buffer);
    free_ppm(ppm_img);

    return 1;
}

int ppm_to_jpg(const char *input_file, const char *output_file, int quality){
    if (quality < 1 || quality > 100) {
        fprintf(stderr, "Quality must be 1-100\n");
        return 1;
    }

    if (!compress_jpeg(input_file, output_file, quality)) {
        fprintf(stderr, "Compression failed\n");
        return 1;
    }
    return 0;
}

/*
Example usage

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input.ppm> <output.jpg> [quality]\n", argv[0]);
        printf("  input.ppm  - Uncompressed PPM image file\n");
        printf("  output.jpg - Output JPEG file\n");
        printf("  quality    - 1-100 (default: 85, higher = better quality)\n");
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int quality = (argc > 3) ? atoi(argv[3]) : 85;

    ppm_to_jpg(input_file, output_file, quality);
}

*/