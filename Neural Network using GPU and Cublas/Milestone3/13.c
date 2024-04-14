#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp.h"
#define max(a, b) ((a) > (b))? (a):(b)
#define NUM_TRAIN_1 10000
#define SIZE 784 // 28*28
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define NUM_VALID 10000
#define OUT_NUM 10
#define e 2.71828
#include<time.h>
#include"/opt/homebrew/Cellar/openblas/0.3.26/include/cblas.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

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

    // Allocate memory for image data
    *data = (unsigned char *)malloc(*numImages * (*rows) * (*cols) * sizeof(unsigned char));

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

void shuffle_array(int arr[], int n) {
  srand(time(NULL)); // Seed the random number generator

  for (int i = n - 1; i > 0; i--) {
    // Generate a random index between 0 and i (inclusive)
    int j = rand() % (i + 1);

    // Swap the element at i with the element at the random index j
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
  }
}


void shuffle(int*rand_num )
{
  //shuffle tranining set for later use.
  srand(time(NULL)); // Seed the random number generator
  for (int i = NUM_TRAIN_1 - 1; i > 0; i--) {
    int j = rand() % (i + 1); // Generate random index within 0 to i
    int temp = rand_num[i];
    rand_num[i] =rand_num[j];
    rand_num[j] = temp;
  }
}


int
main()
{
/*load training and test data*/
int numImages, rows, cols;
unsigned char* data;
unsigned char *labels;
int numLabels;
float * train_image = (float*)malloc(sizeof(float) *NUM_TRAIN * SIZE);
loadMNISTImages("./data/train-images-idx3-ubyte",  &data, &numImages, &rows, &cols);
for(int i=0;i<SIZE*numImages;i++)
    {train_image[i] = (float)data[i];
    train_image[i] = train_image[i]/255;}
printf("%d\n", numImages);
int * train_label = (int*) malloc(sizeof(int) * NUM_TRAIN);
loadMNISTLabels("./data/train-labels-idx1-ubyte", &labels, &numLabels);
for (int i = 0; i < numLabels; i++) 
  train_label[i] = labels[i] ;
printf("%d\n", numImages);
/*float * test_image = (float*)malloc(sizeof(float) * NUM_TEST * SIZE);
loadMNISTImages("./data/t10k-images-idx3-ubyte",  &data, &numImages, &rows, &cols);
for(int i=0;i<SIZE*NUM_TEST;i++)
    test_image[i] = (float) data[i]/255.0f;
printf("%d\n", numImages);
int * test_label = (int*) malloc(sizeof(int) * NUM_TRAIN);
loadMNISTLabels("./data/t10k-labels-idx1-ubyte", &labels, &numLabels);

for (int i = 0; i < numLabels; i++) 
  test_label[i] = (int) labels[i];
printf("%d\n", numImages);*/
int num;
num =15223;
for(int i=0;i<784;i++)
{
    if(i%27==0)
    putchar('\n');
    printf("%1.1f", train_image[num*784+i]);
}
putchar('\n');
printf("%d",labels[num]);
}