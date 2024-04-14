#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#define SIZE 784 // 28*28
#define NUM_TRAIN 60000
#define NUM_TEST 10000
int readInt(FILE *fp) {
    unsigned char buffer[4];
    fread(buffer, sizeof(unsigned char), 4, fp);
    return (buffer[0] << 24) + (buffer[1] << 16) + (buffer[2] << 8) + buffer[3];
}

void loadMNISTImages(const char *filename, unsigned char **data, int *numImages, int *rows, int *cols) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }

    // Read magic number (should be 2051 for images)
    if (readInt(fp) != 2051) {
        fprintf(stderr, "Invalid magic number\n");
        exit(1);
    }

    // Read number of images, rows, and columns
    *numImages = readInt(fp);
    *rows = readInt(fp);
    *cols = readInt(fp);
    int temp;
    // Allocate memory for image data
    *data= (unsigned char *)malloc(*numImages * (*rows) * (*cols) * sizeof(unsigned char));
    // Read image data
    fread(*data, sizeof(unsigned char), *numImages * (*rows) * (*cols), fp);
    fclose(fp);
}


void loadMNISTLabels(const char *filename, unsigned char **labels, int *numLabels) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }

    // Read magic number (should be 2049 for labels)
    if (readInt(fp) != 2049) {
        fprintf(stderr, "Invalid magic number\n");
        exit(1);
    }

    // Read number of labels
    *numLabels = readInt(fp);

    // Allocate memory for label data
    *labels = (unsigned char *)malloc(*numLabels * sizeof(unsigned char));

    // Read label data (each byte represents a label)
    fread(*labels, sizeof(unsigned char), *numLabels, fp);

    fclose(fp);
}


int main() {
    unsigned char *data;
    float * data_1 = malloc(sizeof(float) *NUM_TRAIN * SIZE);
    int numImages, rows, cols, temp;
    loadMNISTImages("./data/train-images-idx3-ubyte",  &data, &numImages, &rows, &cols);
    numImages =1;
    for(int i=0;i<SIZE;i++)
        {
            //temp = data[i] - '0';
            data_1[i] = (float) data[i] / 255.0f;
        }
        
    // Access the data:
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                // data[i * rows * cols + j * cols + k] contains the pixel value
                // You can process or output the pixel value here
                printf("%1.2f ", data_1[i * rows * cols + j * cols + k]);
            }
            printf("\n"); // Separate images with newlines
        }
        printf("\n"); // Separate images with newlines
    }
    
    unsigned char *labels;
    int numLabels, *int_label = (int *)malloc(NUM_TRAIN*sizeof(int));
    loadMNISTLabels("./data/t10k-labels-idx1-ubyte", &labels, &numLabels);
    numLabels = NUM_TRAIN;
    // Access the labels:
    for (int i = 0; i < 10; i++) {
        printf("Label %d: %d\n", i + 1, labels[i]);
    }

    for (int i = 0; i < numLabels; i++) {
        int_label[i] = (int) labels[i] ;

    }
    for (int i = 0; i < 10; i++) {
        printf("Label %d: %d\n", i + 1, int_label[i]);
    }


    free(labels);
    free(data);
    free(data_1);
    return 0;
}





