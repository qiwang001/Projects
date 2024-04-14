#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include "math.h"
#include <time.h>
#include "mnist.h"
// set appropriate path for data
#define TRAIN_IMAGE "./data/train-images-idx3-ubyte"
#define TRAIN_LABEL "./data/train-labels-idx1-ubyte"
#define TEST_IMAGE "./data/t10k-images-idx3-ubyte"
#define TEST_LABEL "./data/t10k-labels-idx1-ubyte"

#define SIZE 784 // 28*28
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define LEN_INFO_IMAGE 4
#define LEN_INFO_LABEL 2
#define OUT_NUM 10

#define MAX_IMAGESIZE 1280
#define MAX_BRIGHTNESS 255
#define MAX_FILENAME 256
#define MAX_NUM_OF_IMAGES 1


int
main()
{
load_mnist();

double * a = (double*) malloc(NUM_TRAIN * SIZE * sizeof(double));
for(int i=0;i<10;i++)
for(int j=0;j<10;j++)
printf("%d", train_image[i][j] == *(&train_image[0][0] +i * SIZE + j));

}