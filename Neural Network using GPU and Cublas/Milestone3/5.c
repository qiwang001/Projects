#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define max(a, b) ((a) > (b))? (a):(b)
#define NUM_TRAIN_1 50000
#define SIZE 784 // 28*28
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define OUT_NUM 10
#define e 2.71828

int 
main()
{
float * A = (float*) malloc(6*sizeof(float));
for(int i=0;i<6;i++)
    A[i]=i + 1;
float * B = (float*) malloc(12*sizeof(float));
for(int i=0;i<12;i++)
    B[i]=i + 1;
float * C = (float*) malloc(8*sizeof(float));
for(int i=0;i<8;i++)
    C[i]=i + 1;
cublasHandle_t handle;
cublasCreate(&handle);
cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, nhidden, nbatch, SIZE
      ,&alpha_1, A, SIZE,  B, nbatch, &beta, C, nbatch);







}