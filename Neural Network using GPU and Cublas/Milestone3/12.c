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
for(int j=0;j<OUT_NUM;j++)
    B[L-1][j] = stddev * ((float) rand() / (RAND_MAX / 2.0f) - 1.0f);
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

void activation_relu(float * Z, float * B, float * A, int nhidden, int nbatch)
{
    int j;
    for(int i=0;i<nbatch*nhidden;i++)
            {j = i%nhidden;
            Z[i] = Z[i] + B[j];}
    
    for(int i=0;i<nbatch*nhidden;i++)
        A[i] = max(Z[i],0);

}

void activation_softmax(float * Z, float * B, float * A, int nbatch)
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

void matrix_add(float*A, float*B, float*C,float alpha,int len)
{
for(int i =0;i<len;i++)
    C[i] = A[i] + alpha * B[i];
}

void relu_prime(float* Z,int len)
{
  for(int i=0;i<len;i++)
    {
      if(Z[i] >0)
        Z[i] = 1;
      else
        Z[i] = 0;
    }
}

void matrix_element_wise_product(float*A, float*B,float*C,int len)
{
  for(int i=0;i<len;i++)
    C[i] = A[i] * B[i];
}

void calculate_loss(float*A, int* y_valid, float*loss,int nbatch, int epo)
{

  int label;
  for(int j = 0; j < nbatch;j++)
    {
      label = (int)y_valid[j];
      loss[epo] += -log(A[j * OUT_NUM + label]); 
    }      

}

void update_Bu(float * Bu, float*deltaL, int height,int nbatch)
{
    int j;
    for(int i=0;i<height*nbatch;i++)
        {
        j = i%height;
        Bu[j] += deltaL[i];
        }
}

void initialize_Z_A_and_delta(float *Z[],float* A[],float*delta[],int L, int nhidden,int nbatch)
{
//Z[0] is not used
Z[0] = NULL;
A[0] = (float*)malloc(SIZE * nbatch * sizeof(float));

for(int i=1;i<L-1;i++)
  {
    Z[i] = (float*)malloc(nbatch * nhidden * sizeof(float));
    A[i]= (float*)malloc(nbatch * nhidden * sizeof(float));
    delta[i]= (float*)malloc(nbatch * nhidden * sizeof(float));
  }

Z[L-1]= (float*)malloc(nbatch * OUT_NUM * sizeof(float));
A[L-1]= (float*)malloc(nbatch * OUT_NUM * sizeof(float));
delta[L-1]= (float*)malloc(nbatch * OUT_NUM * sizeof(float));
}

void initialize_y(int * rand_num, float *y, int* test_label)
{
    for(int i=0;i < NUM_TRAIN_1;i++)
        for(int j=0;j < OUT_NUM;j++)
            if(j==test_label[rand_num[i]])
                y[i * OUT_NUM + j] = 1.0f;
}

void copy_training_data_to_A0(float* A0, float* train_image,int * rand_num,int nbatch,int batch_number)
{
int j;
for(int i=0;i<SIZE*nbatch;i++)
{   
    j = i/SIZE;
    j = rand_num[j + nbatch*batch_number];
    A0[i] = train_image[j*SIZE + i%SIZE];
}
}

void zero_matrix(float*A, int len)
{
for(int i =0;i<len;i++)
    A[i] = 0;
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
int nhidden = 800, nepoch = 5, nbatch = 200;
float alpha = 0.1;
int L = nlayer + 2;

//Wu and Bu are used to update W and B;  Same size as W and B
float * W[L], *B[L], *Wu[L], *Bu[L];

//y: ground truth
float * y=(float*)calloc(OUT_NUM * NUM_TRAIN_1, sizeof(float));

float * loss = (float*)calloc(nepoch,sizeof(float));
initialize_W_and_B_on_host(W, B,L,nhidden);
initialize_W_and_B_on_host(Wu, Bu, L,nhidden);

float *Z[L], *A[L], *delta[L];
initialize_Z_A_and_delta(Z,A,delta,L,nhidden,nbatch);

int *rand_num = (int*) malloc(NUM_TRAIN_1 * sizeof(int));
for(int i=0;i < NUM_TRAIN_1;i++)
  rand_num[i] = i;

float* ones_nbatch = (float*) malloc(sizeof(float) * nbatch);
for(int i=0;i<nbatch;i++)
  ones_nbatch[i] = 1;

float * validation_image = train_image + NUM_TRAIN_1*SIZE;
int * y_valid = train_label + NUM_TRAIN_1;

int niter = NUM_TRAIN_1/nbatch;
double t0 = omp_get_wtime();

for(int epo=0;epo<nepoch;epo++)
{
//shuffle(rand_num);
initialize_y(rand_num, y, test_label);

for(int iter=0;iter<niter;iter++)
{ //copy training data to A[0]
  copy_training_data_to_A0(A[0],train_image, rand_num, nbatch, iter);
  // forward propagation
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nhidden, nbatch, SIZE, alpha_1, W[1], nhidden, A[0], SIZE, beta, Z[1], nhidden);
  activation_relu(Z[1], B[1], A[1], nhidden, nbatch);
    for(int l=2;l<L-1;l++)
    {
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nhidden, nbatch, nhidden,1.0, W[l], nhidden, A[l-1], nhidden, 0.0,Z[l], nbatch);
    activation_relu(Z[l], B[l], A[l], nhidden, nhidden);
    }

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, OUT_NUM, nbatch, nhidden,1.0, W[L-1], OUT_NUM, A[L-2], nhidden, 0.0, Z[L-1], OUT_NUM);
    activation_softmax(Z[L-1],B[L-1],A[L-1],nbatch);
    
    //start to back propagate
    //compute delta[L-1]
    matrix_add(A[L-1], y + OUT_NUM *iter*nbatch, delta[L-1], -1, OUT_NUM * nbatch);

    //compute delta[L-2]
    cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans, nhidden, nbatch, OUT_NUM ,1.0, W[L-1], OUT_NUM, delta[L-1], OUT_NUM, 0.0, delta[L-2],nhidden);
    relu_prime(Z[L-2], nhidden * nbatch);
    matrix_element_wise_product(delta[L-2], Z[L-2],delta[L-2], nhidden * nbatch);

    //following hidden layer delta
    for(int l=L-3;l>=1;l--)
    cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans,nhidden, nbatch,nhidden ,1, W[l+1],nhidden ,delta[l+1],nhidden ,0, delta[l],nhidden);

    //compute update matrix
    zero_matrix(Wu[L-1],OUT_NUM*nhidden);

    //compute Wu[L-1] 
    for(int i=0;i<nbatch;i++)
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, OUT_NUM, nhidden,1,1,delta[L-1] + i * OUT_NUM, OUT_NUM, A[L-2] + i * nhidden, nhidden, 1, Wu[L-1],OUT_NUM);

    //compute Bu[L-1]
    update_Bu(Bu[L-1], delta[L-1], OUT_NUM, nbatch);

    //compute following hidden layer W and B
    for(int l = L-2; l>1;l--)
    {
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, nhidden, nhidden, nbatch ,1, delta[l], nhidden, A[l-1],nbatch, 0.0, Wu[l], nhidden);   
    cblas_sgemm(CblasColMajor, CblasNoTrans,CblasNoTrans, OUT_NUM, 1, nbatch ,1, delta[l], OUT_NUM, ones_nbatch, nbatch, 0, Bu[l], OUT_NUM);
    }

    //compute Wu[1]
    zero_matrix(Wu[1],nhidden*SIZE);
    for(int i=0;i<nbatch;i++)
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, nhidden, SIZE,1,1,delta[1] + i * nhidden, nhidden, A[0] + i * SIZE, SIZE,1,Wu[1],nhidden);

    //compute Bu[1]
    update_Bu(Bu[1],delta[1],nhidden,nbatch);

    //gradient decent
    matrix_add(W[L-1], Wu[L-1], W[L-1], (float)-alpha/nbatch, OUT_NUM * nhidden);
    matrix_add(B[L-1], Bu[L-1], B[L-1], (float)-alpha/nbatch, OUT_NUM);
    if(iter%10==0)
        {   float sum = 0, sum1=0, sum2=0,sum3=0;
            printf("on epochh %d iter %d:\n",epo,iter);
            for(int i=0;i<nhidden*SIZE;i++)
                sum +=W[1][i];
            printf("sum of W1:%f \n", sum);
            for(int i=0;i<nhidden;i++)
                sum2 +=B[1][i];
            printf("sum of B1:%f \n", sum2);
            for(int i=0;i<nhidden*SIZE;i++)
                sum3 +=A[0][i];
             printf("sum of A0:%f \n", sum3);
            for(int i=0;i<nhidden*SIZE;i++)
                sum1 +=A[1][i];
             printf("sum of A1:%f \n", sum1);

            putchar('\n');
        }
        
    matrix_add(W[1], Wu[1], W[1], (float)-alpha/nbatch, SIZE * nhidden);
    matrix_add(B[1], Bu[1], B[1], (float)-alpha/nbatch, nhidden);
}
}
for(int i=0;i<10;i++)
printf("%1.3f ", Wu[2][i]);
putchar('\n');
for(int i=0;i<10;i++)
printf("%1.3f ", Bu[2][i]);
putchar('\n');
for(int i=0;i<10;i++)
printf("%1.3f ", Wu[1][i]);
putchar('\n');
for(int i=0;i<10;i++)
printf("%1.3f ", Bu[1][i]);
putchar('\n');
float sum=0;
for(int i=0;i<SIZE*nbatch;i++)
sum+=Wu[1][i];
printf("%1.3f ", sum);

}