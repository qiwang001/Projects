#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp.h"
#define max(a, b) ((a) > (b))? (a):(b)
#define NUM_TRAIN_1 50000
#define SIZE 784 // 28*28
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define NUM_VALID 10000
#define OUT_NUM 10
#define e 2.71828
#include<time.h>
#include"/opt/homebrew/Cellar/openblas/0.3.26/include/cblas.h"

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
void matrix_element_wise_product(float*A, float*B,float*C,int len)
{
  for(int i=0;i<len;i++)
    C[i] = A[i] * B[i];
}


void initialize_W_and_B_on_host(float* W[], float*B[],int L, int nhidden)
{
W[0]=NULL;
B[0]=NULL;
W[1]= (float*)malloc(SIZE* nhidden * sizeof(float));
B[1]= (float*)malloc(nhidden * sizeof(float));

float stddev = sqrt(2.0/SIZE);
for(int i=0;i<SIZE*nhidden;i++)
  W[1][i] = stddev * ((float) rand() / (RAND_MAX / 2.0f) - 1.0f);

stddev = sqrt(2.0/nhidden);
for(int i=0;i<nhidden;i++)
  B[1][i] = stddev * ((float) rand() / (RAND_MAX / 2.0f) - 1.0f);

for(int i=2;i <L-1 ;i++)
{
  W[i] = (float*)malloc(nhidden* nhidden * sizeof(float));
  for(int j=0;j<nhidden*nhidden;j++)
      W[i][j] = stddev * ((float) rand() / (RAND_MAX / 2.0f) - 1.0f);
  
  B[i] = (float*)malloc(nhidden * sizeof(float));
  for(int j=0;j<nhidden;j++)
      B[i][j] = stddev * ((float) rand() / (RAND_MAX / 2.0f) - 1.0f);
}

W[L-1]=(float*)malloc( OUT_NUM* nhidden * sizeof(float));
  for(int j=0;j<OUT_NUM*nhidden;j++)
      W[L-1][j] = stddev * ((float) rand() / (RAND_MAX / 2.0f) - 1.0f);

B[L-1]=(float*)malloc(OUT_NUM * sizeof(float));
stddev = sqrt(2.0/OUT_NUM);
for(int j=0;j<OUT_NUM;j++)
    B[L-1][j] = stddev * ((float) rand() / (RAND_MAX / 2.0f) - 1.0f);
}

void calculate_loss(float*A, int* y_valid, float*loss,  int nbatch, int epo)
{

  int label;
  for(int j = 0; j < nbatch;j++)
    {
      label = (int)y_valid[j];
      loss[epo] += -log(A[j * OUT_NUM + label]); 
    }      

}

int
main()
{
/*load training and test data*/
unsigned char *data;
float * train_image = (float*)malloc(sizeof(float) *NUM_TRAIN * SIZE);
int numImages, rows, cols;
loadMNISTImages("./data/train-images-idx3-ubyte",  &data, &numImages, &rows, &cols);
for(int i=0;i<SIZE*numImages;i++)
    train_image[i] = (float) data[i] / 255.0f;
unsigned char *labels;
int numLabels;
int * train_label = (int*) malloc(sizeof(int) * NUM_TRAIN);
loadMNISTLabels("./data/t10k-labels-idx1-ubyte", &labels, &numLabels);
for (int i = 0; i < numLabels; i++) 
  train_label[i] = (int) labels[i] ;

float * test_image = (float*)malloc(sizeof(float) * NUM_TEST * SIZE);
loadMNISTImages("./data/t10k-images-idx3-ubyte",  &data, &numImages, &rows, &cols);
for(int i=0;i<SIZE*NUM_TEST;i++)
    test_image[i] = (float) data[i] / 255.0f;

int * test_label = (int*) malloc(sizeof(int) * NUM_TRAIN);
loadMNISTLabels("./data/t10k-labels-idx1-ubyte", &labels, &numLabels);
for (int i = 0; i < numLabels; i++) 
  test_label[i] = (int) labels[i];



int * y_h=(int*)calloc(OUT_NUM * NUM_TRAIN,sizeof(int) );
for(int i=0;i < NUM_TRAIN;i++)
    for(int j=0;j < OUT_NUM;j++)
        if(j==test_label[i])
            y_h[i*OUT_NUM+j] = 1;


int L = 3;
int nhidden =10;
//Wu and Bu are used to update W and B;  Same size as W and B
float * W_h[L], *B_h[L], *Wu_h[L], *Bu_h[L];
float * W[L],   *B[L],   *Wu[L],   *Bu[L];

int*y_valid = train_label + NUM_TRAIN_1;







}