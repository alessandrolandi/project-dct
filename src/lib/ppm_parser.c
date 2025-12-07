#include "parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

//skip whitespace and comments
void skip_comments(FILE *fp) {
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (c == '#') {
            //skip until end of line
            while ((c = fgetc(fp)) != EOF && c != '\n');
        } else if (!isspace(c)) {
            ungetc(c, fp);
            break;
        }
    }
}

//parse ppm for dct
PPMImage* parse_ppm(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }
    
    PPMImage *img = malloc(sizeof(PPMImage));
    if (!img) {
        fclose(fp);
        return NULL;
    }
    
    //read magic val
    char magic[3];
    if (fscanf(fp, "%2s", magic) != 1) {
        fprintf(stderr, "Error: Cannot read magic number\n");
        free(img);
        fclose(fp);
        return NULL;
    }
    
    if (strcmp(magic, "P3") != 0 && strcmp(magic, "P6") != 0) {
        fprintf(stderr, "Error: Invalid PPM format '%s'. Expected P3 or P6.\n", magic);
        free(img);
        fclose(fp);
        return NULL;
    }
    
    int is_binary = (strcmp(magic, "P6") == 0);
    
    //skip comments
    skip_comments(fp);
    
    // Read width and height
    if (fscanf(fp, "%d %d", &img->width, &img->height) != 2) {
        fprintf(stderr, "Error: Cannot read image dimensions\n");
        free(img);
        fclose(fp);
        return NULL;
    }
    
    //skip comments
    skip_comments(fp);
    
    //read max color val
    if (fscanf(fp, "%d", &img->max_val) != 1) {
        fprintf(stderr, "Error: Cannot read max color value\n");
        free(img);
        fclose(fp);
        return NULL;
    }
    
    if (img->max_val > 255 || img->max_val < 1) {
        fprintf(stderr, "Error: Max value %d not supported (must be 1-255)\n", img->max_val);
        free(img);
        fclose(fp);
        return NULL;
    }
    
    //skip single white space after max_val
    fgetc(fp);
    
    //allocate pixel array
    img->pixels = malloc(img->width * img->height * sizeof(Pixel));
    if (!img->pixels) {
        fprintf(stderr, "Error: Cannot allocate memory for pixels\n");
        free(img);
        fclose(fp);
        return NULL;
    }
    
    //read pixel data
    if (is_binary) {
        //P6: binary format
        for (int i = 0; i < img->width * img->height; i++) {
            unsigned char rgb[3];
            if (fread(rgb, 1, 3, fp) != 3) {
                fprintf(stderr, "Error: Unexpected end of file\n");
                free(img->pixels);
                free(img);
                fclose(fp);
                return NULL;
            }
            img->pixels[i].r = (float)rgb[0] - 128.0f;  
            img->pixels[i].g = (float)rgb[1] - 128.0f;
            img->pixels[i].b = (float)rgb[2] - 128.0f;
        }
    } else {
        //P3: ASCII format
        for (int i = 0; i < img->width * img->height; i++) {
            int r, g, b;
            if (fscanf(fp, "%d %d %d", &r, &g, &b) != 3) {
                fprintf(stderr, "Error: Cannot read pixel data\n");
                free(img->pixels);
                free(img);
                fclose(fp);
                return NULL;
            }
            img->pixels[i].r = (float)r - 128.0f;  
            img->pixels[i].g = (float)g - 128.0f;
            img->pixels[i].b = (float)b - 128.0f;
        }
    }
    
    fclose(fp);
    return img;
}

//free ppm image 
void free_ppm(PPMImage *img) {
    if (img) {
        free(img->pixels);
        free(img);
    }
}

//write ppm file from PPMImage
int write_ppm(const char *filename, PPMImage *img, int binary) {
    if (!img || !img->pixels) {
        fprintf(stderr, "Error: Invalid image\n");
        return 0;
    }
    
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        return 0;
    }
    
    //write header
    if (binary) {
        fprintf(fp, "P6\n");
    } else {
        fprintf(fp, "P3\n");
    }
    fprintf(fp, "%d %d\n", img->width, img->height);
    fprintf(fp, "%d\n", img->max_val);
    
    //write pixel data (convert float [-128, 127] back to unsigned char [0, 255])
    if (binary) {
        //P6: Binary format
        for (int i = 0; i < img->width * img->height; i++) {
            //convert [-128, 127] back to [0, 255] and clamp
            float r = img->pixels[i].r + 128.0f;
            float g = img->pixels[i].g + 128.0f;
            float b = img->pixels[i].b + 128.0f;
            
            //clamp to [0, 255]
            if (r < 0.0f) r = 0.0f;
            if (r > 255.0f) r = 255.0f;
            if (g < 0.0f) g = 0.0f;
            if (g > 255.0f) g = 255.0f;
            if (b < 0.0f) b = 0.0f;
            if (b > 255.0f) b = 255.0f;
            
            unsigned char rgb[3] = {
                (unsigned char)r,
                (unsigned char)g,
                (unsigned char)b
            };
            if (fwrite(rgb, 1, 3, fp) != 3) {
                fprintf(stderr, "Error: Failed to write pixel data\n");
                fclose(fp);
                return 0;
            }
        }
    } else {
        //P3: ASCII format
        int count = 0;
        for (int i = 0; i < img->width * img->height; i++) {
            //convert [-128, 127] back to [0, 255] and clamp
            float r = img->pixels[i].r + 128.0f;
            float g = img->pixels[i].g + 128.0f;
            float b = img->pixels[i].b + 128.0f;
            
            //clamp to [0, 255]
            if (r < 0.0f) r = 0.0f;
            if (r > 255.0f) r = 255.0f;
            if (g < 0.0f) g = 0.0f;
            if (g > 255.0f) g = 255.0f;
            if (b < 0.0f) b = 0.0f;
            if (b > 255.0f) b = 255.0f;
            
            fprintf(fp, "%d %d %d ", 
                    (int)r,
                    (int)g,
                    (int)b);
            
            //add newline every 5 pixels for readability
            if (++count % 5 == 0) {
                fprintf(fp, "\n");
            }
        }
    }
    
    fclose(fp);
    return 1;
}

float * flatten(Pixel *pixels, int width, int height){
    float* out = malloc(height * width * 3 * sizeof(float));
    for(int i =  0; i < height * width; i++){
        out[i * 3 + 0] = pixels[i].r;
        out[i * 3  + 1] = pixels[i].g; 
        out[i * 3 + 2] = pixels[i].b;
    }
    return out;
}

/*
int main(int argc, char* argv[] ){

    PPMImage *test = parse_ppm(argv[1]);

    for(int i = 0; i < test->width * test->height; i++ ){
        printf("%f, %f, %f\n", test->pixels[i].r, test->pixels[i].g, test->pixels[i].b);
    }

    free_ppm(test);
}
*/
