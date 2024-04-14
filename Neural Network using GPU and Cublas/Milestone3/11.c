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

int
main()
{ float A[4]={1,2,3,4}, 
B[4] = {1,2,3,4},
C[4] = {1,1,1,1};
cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2,2,2, 1, A, 2, B, 2, 0, C, 2);
  
for(int i=0;i<4;i++)
printf("%f ",C[i]);
}
