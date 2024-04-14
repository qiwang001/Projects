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

int
main()
{
    unsigned char c = 234;
    int f  = c;
    printf("%d", f);
}