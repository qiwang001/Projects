#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define M 5  // Number of rows in A and C
#define N 4  // Number of columns in B and C
#define K 3  // Number of columns in A and rows in B

void initialize_W_and_B_on_GPU(double* W_h[], double* B_h[], double* W[], double*B[],int L, int nhidden)
{
W[0]=NULL;
B[0]=NULL;
cudaMalloc((void **) &W[1], SIZE* nhidden * sizeof(double));
cudaMalloc((void **) &B[1], SIZE* nhidden * sizeof(double));

double stddev = sqrt(2.0/SIZE);
for(int i=0;i<SIZE*nhidden;i++)
  W[1][i] = stddev * ((double) rand() / (RAND_MAX / 2.0f) - 1.0f);

stddev = sqrt(2.0/nhidden);
for(int i=0;i<nhidden;i++)
  B[1][i] = stddev * ((double) rand() / (RAND_MAX / 2.0f) - 1.0f);

for(int i=2;i <L-1 ;i++)
{
  cudaMalloc((void **) &W[i], nhidden* nhidden * sizeof(double));
  for(int j=0;j<nhidden*nhidden;j++)
      W[i][j] = stddev * ((double) rand() / (RAND_MAX / 2.0f) - 1.0f);
  
  cudaMalloc((void **) &B[i], nhidden * sizeof(double));
  for(int j=0;j<nhidden;j++)
      B[i][j] = stddev * ((double) rand() / (RAND_MAX / 2.0f) - 1.0f);

}
cudaMalloc((void **) &W[L-1], OUT_NUM* nhidden * sizeof(double));
  for(int j=0;j<OUT_NUM*nhidden;j++)
      W[L-1][j] = stddev * ((double) rand() / (RAND_MAX / 2.0f) - 1.0f);

cudaMalloc((void **) &B[L-1], OUT_NUM * sizeof(double));
stddev = sqrt(2.0/OUT_NUM);
for(int j=0;j<OUT_NUM;j++)
    B[L-1][j] = stddev * ((double) rand() / (RAND_MAX / 2.0f) - 1.0f);
}


int
main()
{
cublasHandle_t handle;
cublasCreate(&handle);
double * a_h = (double*) malloc(M*K * sizeof(double));
double * b_h = (double*) malloc(K*N * sizeof(double));
double * c_h = (double*) malloc(M*N * sizeof(double));
for(int i=0;i<M*K;i++)
    a_h[i] = i+1;

for(int i=0;i<K*N;i++)
    b_h[i] = i+1;

for(int i=0;i<M*N;i++)
    c_h[i] = 101+1;

double * a;
cudaMalloc((void**) &a,M*K * sizeof(double));
double * b;
cudaMalloc((void**) &b,K*N * sizeof(double));
double * c;
cudaMalloc((void**) &c,M*N * sizeof(double));

cudaMemcpy(a, a_h,M*K*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(b, b_h,K*N*sizeof(double), cudaMemcpyHostToDevice);
double alpha = 1;
double beta = 0;
cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,M, N, K,&alpha, a, M, b, K,& beta,c,M);
beta = 0.1;
cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,M, N, K,&alpha, a, M, b, K,& beta,c,M);
cudaMemcpy(c_h, c,M*N*sizeof(double), cudaMemcpyDeviceToHost);
for(int i=0;i<8;i++)
    printf("%f ", c_h[i]);
cublasDestroy(handle);
return 0;
}
