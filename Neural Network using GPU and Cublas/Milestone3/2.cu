#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "omp.h"
#define max(a, b) ((a) > (b))? (a):(b)
#define NUM_TRAIN_1 50000
#define SIZE 784 // 28*28
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define NUM_VALID 10000
#define OUT_NUM 10
#define e 2.71828
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

void initialize_W_and_B_on_GPU(float* W_h[], float*B_h[],float* W[], float*B[],int L, int nhidden)
{
W[0] = NULL;
B[0] = NULL; 
cudaMalloc((void **) &W[1], SIZE*nhidden*sizeof(float));
cudaMemcpy(W[1],W_h[1],SIZE*nhidden*sizeof(float),cudaMemcpyHostToDevice);
cudaMalloc((void **) &B[1], nhidden*sizeof(float));
cudaMemcpy(B[1],B_h[1],nhidden*sizeof(float),cudaMemcpyHostToDevice);

for(int i=2;i<L-1;i++)
{
cudaMalloc((void **) &W[i], nhidden * nhidden * sizeof(float));
cudaMemcpy(W[i], W_h[i], nhidden * nhidden * sizeof(float),cudaMemcpyHostToDevice);
cudaMalloc((void **) &B[i], nhidden * sizeof(float));
cudaMemcpy(B[i], B_h[i], nhidden * nhidden * sizeof(float),cudaMemcpyHostToDevice);
}

cudaMalloc((void **) &W[L-1], OUT_NUM * nhidden * sizeof(float));
cudaMemcpy(W[L-1], W_h[L-1], nhidden * nhidden * sizeof(float),cudaMemcpyHostToDevice);
cudaMalloc((void **) &B[L-1], OUT_NUM * sizeof(float));
cudaMemcpy(B[L-1], B_h[L-1], OUT_NUM * sizeof(float),cudaMemcpyHostToDevice);
}

void initialize_Z_A_and_delta_on_GPU(float *Z[],float* A[],float*delta[],int L, int nhidden,int nbatch)
{
//Z[0] is not used
Z[0] = NULL;
cudaMalloc((void**)&A[0], SIZE * nbatch * sizeof(float));
for(int i=1;i<L-1;i++)
  {
    cudaMalloc((void**)&Z[i], nbatch * nhidden * sizeof(float));
    cudaMalloc((void**)&A[i], nbatch * nhidden * sizeof(float));
    cudaMalloc((void**)&delta[i], nbatch * nhidden * sizeof(float));
  }

cudaMalloc((void**)&Z[L-1], nbatch * OUT_NUM * sizeof(float));
cudaMalloc((void**)&A[L-1], nbatch * OUT_NUM * sizeof(float));
cudaMalloc((void**)&delta[L-1], nbatch * OUT_NUM * sizeof(float));
}

void copy_training_data_to_GPU(float * train_image_on_host, float* train_image_on_gpu, float*train_label_on_gpu ,float*train_label_on_host)
{
  cudaMalloc((void**)&train_image_on_gpu, NUM_TRAIN_1 * SIZE * sizeof(float));
  cudaMalloc((void**)&train_label_on_gpu, NUM_TRAIN_1  * sizeof(int));
  cudaMemcpy(train_image_on_gpu, train_image_on_host, NUM_TRAIN_1 * SIZE * sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(train_label_on_gpu, train_label_on_host, NUM_TRAIN_1  * sizeof(int),cudaMemcpyHostToDevice);
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

__global__ void activation_relu(float * Z, float * B, float * A, int nhidden, int nbatch)
{
  for(int i=0;i<nhidden;i++)
    for(int j=0;j<nbatch;j++)
     {
      Z[i * nbatch + j] = Z[i * nbatch + j] + B[i];
      A[i * nbatch + j] = max(Z[i * nbatch + j], 0);
     }
}

__global__ void activation_softmax(float * Z, float * B, float * A, int nbatch)
{
  float *sum = (float*)malloc(sizeof(float) * nbatch);
  for(int j=0;j<nbatch;j++)
    for(int i=0;i<OUT_NUM;i++)    
       Z[i + j * OUT_NUM] = Z[i + j * OUT_NUM] + B[i];
       
  for(int j=0;j<nbatch*OUT_NUM;j++)
        Z[j] = pow(e,Z[j]);

  for(int j=0;j<nbatch;j++)
    for(int i=0;i<OUT_NUM;i++)
        sum[j] += Z[j*OUT_NUM + i]; 

for(int j=0;j<nbatch;j++)
    for(int i=0;i<OUT_NUM;i++)
    A[j*OUT_NUM+i] = Z[j*OUT_NUM + i]/sum[j];

}
//C = A + alpha * B
__global__ void matrix_add(float*A, float*B, float*C,float alpha,int len)
{
for(int i =0;i<len;i++)
    C[i] = A[i] + alpha*B[i];
}

__global__ void zero_matrix(float*A, int len)
{
for(int i =0;i<len;i++)
    A[i] = 0;
}

__global__ void relu_prime(float* Z,int len)
{
  for(int i=0;i<len;i++)
    {
      if(Z[i] >0)
        Z[i] = 1;
      else
        Z[i] = 0;
    }
}

__global__ void matrix_element_wise_product(float*A, float*B,float*C,int len)
{
  for(int i=0;i<len;i++)
    C[i] = A[i] * B[i];
}

__global__ void calculate_loss(float*A, int* y_valid, float*loss, int batch_idx, int nbatch, int epo)
{

  int label;
  for(int j = batch_idx * nbatch; j < nbatch * (batch_idx + 1);j++)
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

float alpha_1 = 1;
float beta = 0;
int nlayer = 1;
int nhidden = 800, nepoch = 10, nbatch = 200;
float alpha = 0.1;
int L = nlayer + 2;
int niter = NUM_TRAIN_1/nbatch;
int zero = 0, one = 1;

//Wu and Bu are used to update W and B;  Same size as W and B
float * W_h[L], *B_h[L], *Wu_h[L], *Bu_h[L];
float * W[L],   *B[L],   *Wu[L],   *Bu[L];

//y: ground truth
float * y;
cudaMalloc((void**)&y, sizeof(float)*nbatch*OUT_NUM);
float * loss;
cudaMalloc((void**)&loss, sizeof(float)*nepoch);
zero_matrix<<<1,1>>>(loss, nepoch);


initialize_W_and_B_on_host(W_h, B_h,L,nhidden);
initialize_W_and_B_on_host(Wu_h, Bu_h, L,nhidden);
initialize_W_and_B_on_GPU(W_h, B_h,W,B,L,nhidden);
initialize_W_and_B_on_GPU(Wu_h,Bu_h,Wu,Bu,L,nhidden);
float *Z[L], *A[L], *delta[L];
initialize_Z_A_and_delta_on_GPU(Z,A,delta,L,nhidden,nbatch);
//copy_training_data_to_GPU(&train_image[0][0], train_image_on_gpu, train_label_on_gpu);
/* start to train on GPU*/
cublasHandle_t handle;
cublasCreate(&handle);
int *rand_num = (int*) malloc(NUM_TRAIN_1 * sizeof(int));
for(int i=0;i < NUM_TRAIN_1;i++)
  rand_num[i] = i;

float* ones_nbatch = (float*) malloc(sizeof(float) * nbatch);
for(int i=0;i<nbatch;i++)
  ones_nbatch[i] = 1;

float * validation_image;
int*y_valid;
cudaMalloc((void**)&validation_image, sizeof(float) * NUM_VALID * SIZE);
cudaMalloc((void**)&y_valid, sizeof(float) * NUM_VALID);

cudaMemcpy(validation_image, train_image + SIZE*NUM_TRAIN_1, NUM_VALID * SIZE * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(y_valid, train_label + NUM_TRAIN_1, NUM_VALID * sizeof(float), cudaMemcpyHostToDevice);

double t0 = omp_get_wtime();
for(int epo=0;epo<nepoch;epo++)
{
for(int iter=0;iter<niter;iter++)
{
  shuffle(rand_num);
  /*copy training data to A[0]*/
  for(int i=0;i<nbatch;i++)
      cudaMemcpy((A[0]+i*SIZE), train_image + rand_num[iter*nbatch+i]*SIZE, SIZE*sizeof(float), cudaMemcpyHostToDevice);
  
  /*copy labels into y*/  
  for(int i=0;i<OUT_NUM;i++)
    for(int j=0;j< nbatch;j++)
        {
        if(train_label[rand_num[iter*nbatch+j]] ==  i )
            cudaMemcpy(&y[i * nbatch + j], &one, sizeof(int), cudaMemcpyHostToDevice);
        else
            cudaMemcpy(&y[i * nbatch + j], &zero, sizeof(int), cudaMemcpyHostToDevice);
        }

      /* forward propagation*/
      cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, nhidden, nbatch, SIZE,&alpha_1, W[1], nhidden,  A[0], SIZE, &beta, Z[1], nhidden);
      activation_relu<<<1, 1>>>(Z[1], B[1], A[1], nhidden, nbatch);

      for(int l=2;l<L-1;l++)
      {
      cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, nhidden, nbatch, nhidden,&alpha_1, W[l], nhidden, A[l-1], nhidden, &beta,Z[l], nbatch);
      activation_relu<<<1, 1>>>(Z[l], B[l], A[l], nhidden, nhidden);
      }

      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, OUT_NUM, nbatch, nhidden,&alpha_1, W[L-1], OUT_NUM, A[L-2], nhidden, &beta, Z[L-1], OUT_NUM);
      activation_softmax<<<1,1>>>(Z[L-1],B[L-1],A[L-1],nbatch);

    //start to back propagate
    //compute delta[L-1]
    matrix_add<<<1,1>>>(A[L-1], y, delta[L-1], -1, OUT_NUM * nbatch);
    
    //compute delta[L-2]
    cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N, nhidden, nbatch, OUT_NUM ,&alpha_1, W[L-1], OUT_NUM, delta[L-1], OUT_NUM, &beta, delta[L-2],nhidden );
    relu_prime<<<1,1>>>(Z[L-2], nhidden * nbatch);
    matrix_element_wise_product<<<1,1>>>(delta[L-2], Z[L-2],delta[L-2], nhidden * nbatch);
    
    //following hidden layer delta
    for(int l=L-3;l>=1;l--)
    cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,nhidden, nbatch,nhidden ,&alpha_1, W[l+1],nhidden , delta[l+1],nhidden , &beta,delta[l],nhidden);
    
   //compute update matrix
   //compute Wu[L-1] 
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, OUT_NUM, nhidden, nbatch ,&alpha_1, delta[L-1], OUT_NUM, A[L-2],nhidden, &beta, Wu[L-1], OUT_NUM);
    
    //compute Bu[L-1]
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, OUT_NUM, 1, nbatch ,&alpha_1, delta[L-1], OUT_NUM, ones_nbatch, nbatch, &beta, Bu[L-1], OUT_NUM);

    //compute following hidden layer W and B
    for(int l = L-2; l>1;l--)
        {cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, nhidden, nhidden, nbatch ,&alpha_1, delta[l], nhidden, A[l-1],nbatch, &beta, Wu[l], nhidden);   
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, OUT_NUM, 1, nbatch ,&alpha_1, delta[l], OUT_NUM, ones_nbatch, nbatch, &beta, Bu[l], OUT_NUM);
        }

    //compute W[1] and B[1]
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, nhidden, SIZE, nbatch 
    ,&alpha_1, delta[1], nhidden, A[0], SIZE, &beta, Wu[1], nhidden);

    //compute Bu[1]
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, OUT_NUM, 1, nbatch ,&alpha_1, delta[1], OUT_NUM, ones_nbatch, nbatch, &beta, Bu[1], OUT_NUM);

    //gradient decent
    matrix_add<<<1,1>>>(W[L-1], Wu[L-1], W[L-1], (float)-alpha/nbatch, OUT_NUM * nhidden);
    matrix_add<<<1,1>>>(B[L-1], Bu[L-1], B[L-1], (float)-alpha/nbatch, OUT_NUM);
        
    //compute following hidden layer W and B
    for(int l = L-2; l>1;l--)
        { matrix_add<<<1,1>>>(W[l], Wu[l], W[l], (float)-alpha/nbatch, nhidden * nhidden);
          matrix_add<<<1,1>>>(B[l], Bu[l], B[l], (float)-alpha/nbatch, nhidden);
        }
    matrix_add<<<1,1>>>(W[L-1], Wu[L-1], W[L-1], (float)-alpha/nbatch, SIZE * nhidden);
    matrix_add<<<1,1>>>(B[L-1], Bu[L-1], B[L-1], (float)-alpha/nbatch, nhidden);
}//one batch

//calculate loss using validation set
int temp = NUM_VALID/nbatch;
float * ptr =NULL;

for(int i=0;i < temp;i++)
  {
  
  ptr = validation_image + i * SIZE * nbatch;
  cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, nhidden, nbatch, SIZE, &alpha_1, W[1], nhidden, ptr, SIZE, &beta, Z[1], nhidden);
  activation_relu<<<1, 1>>>(Z[1], B[1], A[1], nhidden, nbatch);

  /*compute middle hidden layer*/

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, OUT_NUM, nbatch, nhidden, &alpha_1, W[L-1], OUT_NUM, A[L-2], nhidden, &beta, Z[L-1], OUT_NUM);
  activation_softmax<<<1,1>>>(Z[L-1],B[L-1],A[L-1],nbatch);
  calculate_loss<<<1,1>>>(A[L-1], y_valid, loss, i, nbatch, epo);
  }

}//training end
double t1 = omp_get_wtime();
float * loss_h = (float*) malloc(sizeof(float) * nepoch);
float * AL = (float*) malloc(10*sizeof(float));
cudaMemcpy(loss_h, loss,sizeof(float) * nepoch,cudaMemcpyDeviceToHost );
cudaMemcpy(W_h[1], W_h,sizeof(float) * nhidden*SIZE,cudaMemcpyDeviceToHost);
cudaMemcpy(AL, A[L-1],sizeof(float) * 10,cudaMemcpyDeviceToHost);
printf("time used to train NN:%f\n ", t1-t0);
for(int i=0;i<nepoch;i++)
printf("%f\n ", loss_h[i]);
for(int i=0;i<nepoch;i++)
printf("%f\n ", W_h[1][i]);
for(int i=0;i<10;i++)
printf("%f\n ", AL[i]);
cublasDestroy(handle);
return 0;
}

  












