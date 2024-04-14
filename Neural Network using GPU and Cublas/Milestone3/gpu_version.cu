#define e 2.71828
#include "mnist.h"
#include "omp.h"
#include<time.h>
#include "math.h"
#include <stdio.h>
//#include"/opt/homebrew/Cellar/openblas/0.3.26/include/cblas.h"
//#include<cblas.h>
#include <cuda.h>
#include "cublas_v2.h"

void initialize_W_and_B_on_GPU(double* W[], double*B[],int L, int nhidden)
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

//
void initialize_update_matrix_on_GPU(double* uW[], double*uB[],int L, int nhidden)
{
uW[0]=NULL;
uB[0]=NULL;
cudaMalloc((void **) &uW[1], SIZE* nhidden * sizeof(double));
cudaMalloc((void **) &uB[1], nhidden * sizeof(double));

for(int i=2;i <L-1 ;i++)
{
  cudaMalloc((void **) &uW[i], nhidden* nhidden * sizeof(double));
  cudaMalloc((void **) &uB[i], nhidden * sizeof(double));
}

cudaMalloc((void **) &uW[L-1], OUT_NUM* nhidden * sizeof(double));
cudaMalloc((void **) &uB[L-1], OUT_NUM * sizeof(double));

}

void initialize_W_and_B_on_host(double* W[], double*B[],int L, int nhidden)
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


__global__ void train(double *W[], double*B[], double*uW[], double*uB[],double*uW_acc[],
double*uB_acc[],double*a[], double *Z[],int L, int nhidden, int nbatch,int*rand_num,double alpha,int nepoch)
{
for(int i=0;i<NUM_TRAIN;i++)
    rand_num[i]=i;

cublasHandle_t handle;
cublasCreate(&handle);

//start to forward_propagation
int niter = NUM_TRAIN/nbatch;
for(int epo =0; epo<nepoch;epo++)
{
//shuffle tranining set for later use.
  srand(time(NULL)); // Seed the random number generator
  for (int i = NUM_TRAIN - 1; i > 0; i--) {
    int j = rand() % (i + 1); // Generate random index within 0 to i
    int temp = rand_num[i];
    rand_num[i] =rand_num[j];
    rand_num[j] = temp;
  }
double debug;
for(int iter=0;iter<niter;iter++)
{   
    printf("training on epoch %d batch%d\n", epo, iter);
    debug=0;
    for(int i=0;i<nhidden*nhidden;i++)
      debug+=W[1][i];
    //printf("W[1] sum %f\n", debug);
    //set update matrix values to 0
    for(int i=0;i<nhidden*SIZE;i++)
      uW_acc[1][i] = 0;
    for(int i=0;i<nhidden;i++)
      uB_acc[1][i] = 0;

    for(int l=2;l <L-1 ;l++)
    {
      for(int i=0;i<nhidden*nhidden;i++)
        uW_acc[l][i] = 0;  
      for(int i=0;i<nhidden;i++)  
        uB_acc[l][i] =0;
    }

    for(int i=0;i<OUT_NUM*nhidden;i++)
        uW_acc[L-1][i] = 0;
    for(int i=0;i<OUT_NUM;i++)
        uB_acc[L-1][i] = 0;


    //start training on a batch
    //put a batch of image values into a[0]
    int img;
    for(int i=0;i<SIZE;i++)
      for(int j=0;j< nbatch;j++)
        {
        img=rand_num[iter*nbatch + j];
        a[0][i * nbatch + j] = train_image[img][i];
        }
    int lab;
    //put a batch of label into y
    for(int i=0;i<OUT_NUM;i++)
      for(int j=0;j<nbatch;j++)
        {
          lab = rand_num[iter*nbatch+j];
          y[i*nbatch+j] = (train_label[lab]==i)? 1:0;
        }  
   
    //forward propagation using blas
    // compute first hidden layer
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,nhidden, nbatch, SIZE,
      1.0, W[1], SIZE, a[0], nhidden, 0.0, Z[1], nbatch);

      //printf("%s \n", "Z[1]");
      for(int i=0;i<nhidden;i++)
        //printf("%f ", Z[1][i]);
        putchar('\n');
      for(int i =0;i<nhidden;i++)
        for(int j=0;j<nbatch;j++)
          a[1][i*nbatch+j] = ((Z[1][i*nbatch+j] + B[1][i])>0? 1.0:0.0);

      //computing following hidden layer's activations
      for(int l=2;l<=L-2;l++)
          {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,nhidden, nbatch, nhidden, 
            1.0, W[l], nhidden, a[l-1], nbatch, 0.0, Z[l], nbatch);
            for(int i =0;i<nhidden;i++)
              for(int j=0;j<nbatch;j++)
                a[l][i*nbatch+j] = ((Z[l][i*nbatch+j] + B[l][i] > 0) ? 1.0:0.0);  
          }        

        //compute the output layer using softmax activation
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,OUT_NUM, nbatch, nhidden, 
        1.0, W[L-1], nhidden, a[L-2], nbatch, 0.0, Z[L-1], nbatch);
        printf("%s\n ", "Z[2]");
        for(int i=0;i<OUT_NUM;i++)
            printf("%f ", Z[2][i]);
        putchar('\n');
        double sum1= 0;
        for(int i=0; i<nbatch*OUT_NUM;i++)
          {
            Z[L-1][i] = pow(e, Z[L-1][i]);
            sum1 += Z[L-1][i];
          }
        
        for(int i=0; i<nbatch*OUT_NUM;i++)
            a[L-1][i] = Z[L-1][i]/sum1;

        //Backpropagate 
        //compute output layer error
        //using cross entrophy loss
        printf("%s ", "start backpropagete\n");
        for(int i = 0;i < nbatch*OUT_NUM; i++)
            delta[L-1][i] = a[L-1][i] - y[i];
        printf("%s","delta2\n");
        for(int i=0;i<10;i++)
            printf("%f ", delta[2][i]);
        putchar('\n');
        //compute hidden layer errors 
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,OUT_NUM, nbatch, nhidden,
        1.0, W[L-1],OUT_NUM, delta[L-1], nbatch,0.0,delta[L-2],nbatch);

        printf("%s","delta1:\n");
        for(int i=0;i<10;i++)
            printf("%f ", delta[1][i]);
        putchar('\n');
        for(int i=0;i<nhidden*nbatch;i++)
           delta[L-2][i] = delta[L-2][i] * ((Z[L-2][i] > 0)?1.0:0.0);
        printf("%s","delta1:\n");
        for(int i=0;i<10;i++)
            printf("%f ", delta[1][i]);
        putchar('\n');
        //compute following hidden layer error
        for(int l=L-3;l>=1;l--)
        {
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,nhidden,nbatch, nhidden, 
            1, WT,nhidden, delta[l+1], nbatch,0.0,delta[l],nbatch);

            for(int i=0;i<nhidden*nbatch;i++)
                delta[L-2][i] = delta[L-2][i] * relu_prime(Z[L-2][i]);
            
        }
        
        //compute values used to update W and B
        //last layer
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,OUT_NUM,nhidden,nbatch, 1, 
        delta[L-1], nbatch, a[L-2], nhidden,0.0,uW[L-1],nhidden);  
        for(int i=0;i<OUT_NUM*nhidden;i++)
          uW_acc[L-1][i] += uW[L-1][i];
        for(int i=0;i<OUT_NUM;i++)
          uB_acc[L-1][i] += delta[L-1][i];
        for(int l=L-2;l>1;l--)
        {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,nhidden,nhidden,nbatch, 1, 
        delta[l], nbatch, a[l-1], nhidden,0.0,uW[l],nhidden);
        for(int i=0;i<nhidden*nhidden;i++)
          uW_acc[l][i] += uW[l][i];
        for(int i=0;i<nhidden;i++)
          uB_acc[l][i] += delta[l][i];
        }

        //computer layer 1
        printf("%s", "Ta[0]:\n");
        for(int i=100;i<200;i++)
          printf("%f ", Ta[i]);
          putchar('\n');
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,SIZE,nhidden,nbatch, 1, 
        delta[1], nbatch, a[0], nhidden,0.0,uW[1],nhidden);   
        for(int i=0;i<SIZE*nhidden;i++)
          uW_acc[1][i] += uW[1][i];
        for(int i=0;i<SIZE;i++)
          uB_acc[1][i] += delta[1][i];
      printf("%s:", "uW[1]:\n");
      for(int i=0;i<100;i++)
      printf("%f ", uW[1][i]);
      putchar('\n');
  
 //for(int i = 0;i<nhidden*nhidden;i++)
   // printf("%f\n ", W[1][i]);
    //gradient decent
    printf("%s","gradient decent.");
    for(int i=0;i<SIZE*nhidden;i++)
      W[1][i] = W[1][i] - uW_acc[1][i]*alpha/nbatch;
    for(int i=0;i<nhidden;i++)
      B[1][i] = B[1][i] - uB_acc[1][i]*alpha/nbatch;

    for(int l=L-2;l>1;l--)
    {
      for(int i=0;i< nhidden * nhidden;i++)
      W[l][i] = W[l][i] - (alpha/nbatch) * uW_acc[l][i];
      for(int i=0;i<nhidden;i++)
      B[l][i] = B[l][i] - (alpha/nbatch)*uB_acc[l][i];
    }

    for(int i=0;i<OUT_NUM*nhidden;i++)
      W[L-1][i] = W[L-1][i] - uW_acc[L-1][i]*alpha/nbatch;
    for(int i=0;i<OUT_NUM;i++)
      B[L-1][i] = B[L-1][i] - uB_acc[L-1][i]*alpha/nbatch;

    printf("%s ", "W[1]:\n");
    for(int i=0;i<100;i++)
      printf("%f ", W[1][i]);
    putchar('\n');
  }

  // one batch end
}// one epoch end
// traininig end

return
}

int
main()
{
//load training data
load_mnist();
srand(time(NULL));
unsigned int seed = time(NULL);
/*int nlayer = atoi(argv[1]), nhidden = atoi(argv[2]), nepoch = atoi(argv[3]), nbatch = atoi(argv[4]);
double alpha = atof(argv[5]);
*/
int nlayer = 1;
int nhidden = 800, nepoch = 3, nbatch = 256;

double alpha = 0.01;
int L = nlayer + 2;
int *y = (int*)malloc(sizeof(int) * OUT_NUM * nbatch);

//parameters on GPU
double* W_gpu_temp[L];
double* B_gpu_temp[L];
double* W_gpu[L];
double* B_gpu[L];
cudaMalloc((void **) &W_gpu[0], L*sizeof(double*));
cudaMalloc((void **) &B_gpu[0], L*sizeof(double*));

//allocate memory for W and B in GPU, and kaiming initialize it
initialize_W_and_B_on_GPU(W_gpu_temp, B_gpu_temp, L, nhidden);

//copy the  address of each Wi and Bi to GPU
cudaMemcpy(&W_gpu[0],&W_gpu_temp[0],L*sizeof(double*),cudaMemcpyHostToDevice);
cudaMemcpy(&B_gpu[0],&B_gpu_temp[0],L*sizeof(double*),cudaMemcpyHostToDevice);

//do the same for uW,uB, uW_acc, uB_acc
//parameters on GPU
double* uW_gpu_temp[L];
double* uB_gpu_temp[L];
double* uW_gpu[L];
double* uB_gpu[L];
cudaMalloc((void **) &uW_gpu[0], L*sizeof(double*));
cudaMalloc((void **) &uB_gpu[0], L*sizeof(double*));

//allocate memory for W and B in GPU, and kaiming initialize it
initialize_W_and_B_on_GPU(uW_gpu_temp, uB_gpu_temp, L, nhidden);

//copy the  address of each Wi and Bi to GPU
cudaMemcpy(&uW_gpu[0],&uW_gpu_temp[0],L*sizeof(double*),cudaMemcpyHostToDevice);
cudaMemcpy(&uB_gpu[0],&uB_gpu_temp[0],L*sizeof(double*),cudaMemcpyHostToDevice);

//parameters on GPU
double* uW_acc_gpu_temp[L];
double* uB_acc_gpu_temp[L];
double* uW_acc_gpu[L];
double* uB_acc_gpu[L];
cudaMalloc((void **) &uW_acc_gpu[0], L*sizeof(double*));
cudaMalloc((void **) &uB_acc_gpu[0], L*sizeof(double*));

//allocate memory for W and B in GPU, and kaiming initialize it
initialize_W_and_B_on_GPU(uW_acc_gpu_temp, uB_acc_gpu_temp, L, nhidden);

//copy the  address of each Wi and Bi to GPU
cudaMemcpy(&uW_acc_gpu[0],&uW_acc_gpu_temp[0],L*sizeof(double*),cudaMemcpyHostToDevice);
cudaMemcpy(&uB_acc_gpu[0],&uB_acc_gpu_temp[0],L*sizeof(double*),cudaMemcpyHostToDevice);

//allocate memory for traininig data on GPU
double *train_imageG_temp[NUM_TRAIN];
int* train_labelG_temp[NUM_TRAIN];
for(int i=0;i<NUM_TRAIN;i++)
    {
      cudaMalloc((void **) &train_imageG_temp[i], SIZE * sizeof(double));
      cudaMalloc((void **) &train_labelG_temp[i], OUT_NUM * sizeof(int));
    }

//copy training data from host to device
for(int i=0;i<NUM_TRAIN;i++)
  {
  cudaMemcpy(train_imageG_temp[i], &train_image[i][0],SIZE*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(train_labelG_temp[i], &train_label[i][0],sizeof(int),cudaMemcpyHostToDevice);
  }

//allocate memory for the address of training data
double *train_imageGPU;
int * train_labelGPU;
cudaMalloc((void **) &train_imageGPU, NUM_TRAIN * sizeof(double*));
cudaMalloc((void **) &train_labelGPU, NUM_TRAIN * sizeof(int*));



//allocate memory for W and B on host. Later receive W and B from GPU
//parameters on host; W[0] and B[0] are not used.
double* W[L];
double* B[L];
double sum=0,t=0,debug=0;
initialize_W_and_B_on_host(W,B,L,nhidden);


//allocate activation matrix on GPU
//activatin matrix
double * a_temp[L];
double * a;
cudaMalloc((void **) &a_temp[0], nbatch * SIZE * sizeof(double));

for(int i=1;i<L-1;i++)
  cudaMalloc((void **) &a_temp[i], nbatch * nhidden * sizeof(double));
cudaMalloc((void **) &a_temp[L-1], nbatch * OUT_NUM * sizeof(double));

//create a array in GPU, each element holds address of a[i]
cudaMalloc((void **) &a, L * sizeof(double*));
//copy address of a[i] to gpu
cudaMemcpy(a, a_temp[0], L*sizeof(double*),cudaMemcpyHostToDevice);


//allocate memory for pre action matrix Z;Z[0] is not used
double * Z_temp[L];
double * Z;
Z_temp[0]=NULL;
for(int i=1;i<L-1;i++)
  cudaMalloc((void **) &Z_temp[i], nhidden * nbatch *  sizeof(double));
cudaMalloc((void **) &Z_temp[L-1], OUT_NUM * nbatch * sizeof(double));

//create a array in GPU, each element holds address of Z[i]
cudaMalloc((void **) &Z, L * sizeof(double*));
//copy address of a[i] to gpu
cudaMemcpy( Z,Z_temp[0], L*sizeof(double*),cudaMemcpyHostToDevice);


//errors for hidden layers
//delta[0] is not used
double *delta_temp[L];
delta_temp[0]=NULL;
double *delta;
for(int i=1;i<L-1;i++)
  cudaMalloc((void **) &delta_temp[i], nhidden * nbatch *  sizeof(double));
cudaMalloc((void **) &delta_temp[L-1], OUT_NUM * nbatch * sizeof(double));

//create a array in GPU, each element holds address of delta[i]
cudaMalloc((void **) &delta, L * sizeof(double*));
//copy address of a[i] to gpu
cudaMemcpy( delta,delta_temp[0], L*sizeof(double*),cudaMemcpyHostToDevice);

//array used to support shuffle training data
int *rand_num;
cudaMalloc((void **) &rand_num, NUM_TRAIN *  sizeof(int));


//start to train on GPU
train<<<nblocks, nthreads>>>( W, B, uW, uB,uW_acc_gpu,uB_acc_gpu,a,Z,
L, nhidden, nbatch,rand_num, alpha, nepoch);
//move W and B back to host
cudaMemcpy(W[1],W_gpu[1],nhidden * SIZE*sizeof(double),cudaMemcpyDeviceToHost);
cudaMemcpy(B[1],B_gpu[1],nhidden * SIZE*sizeof(double),cudaMemcpyDeviceToHost);
for(int l=2;l<=L-2;i++)
{
cudaMemcpy(W[l],W_gpu[l],nhidden * nhidden*sizeof(double),cudaMemcpyDeviceToHost);
cudaMemcpy(B[l],B_gpu[l],nhidden * nhidden*sizeof(double),cudaMemcpyDeviceToHost);

}
cudaMemcpy(W[L-1],W_gpu[L-1],nhidden * OUT_NUM*sizeof(double),cudaMemcpyDeviceToHost);
cudaMemcpy(B[L-1],B_gpu[L-1],nhidden * OUT_NUM*sizeof(double),cudaMemcpyDeviceToHost);


// start testing
int hits=0, guess=0, total;
double max= -INFINITY;;
niter = NUM_TEST/nbatch;
total = nbatch * niter;

//start training on a batch
//put a batch of image values into a[0]
int img;
for(int iter=0;iter<niter;iter++)
{
for(int i=0;i<SIZE;i++)
  for(int j=0;j< nbatch;j++)
    {
    img=iter*nbatch + j;
    a[0][i * nbatch + j] = test_image[img][i];
    }

//forward propagation using blas
// compute first hidden layer
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            nhidden, nbatch, SIZE, 1.0, W[1], SIZE, a[0], 
            nhidden, 0.0, Z[1], nbatch);

for(int i =0;i<nhidden;i++)
  for(int j=0;j<nbatch;j++)
    a[1][i*nbatch+j] = relu(Z[1][i*nbatch+j] + B[1][i]);

//computing following hidden layer's activations
for(int l=2;l<=L-2;l++)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        nhidden, nbatch, nhidden, 1.0, W[l], nhidden, a[l-1], 
        nbatch, 0.0, Z[l], nbatch);
        for(int i =0;i<nhidden;i++)
          for(int j=0;j<nbatch;j++)
            a[l][i*nbatch+j] = relu(Z[l][i*nbatch+j] + B[l][i]);  
    } 

//compute the output layer using softmax activation
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        OUT_NUM, nbatch, nhidden, 1.0, W[L-1], nhidden, a[L-2], 
        nbatch, 0.0, Z[L-1], nbatch);

//compute hits number
for(int j=0;j<nbatch;j++)
 { for(int i=0;i<OUT_NUM;i++)
      {if(Z[L-1][i * nbatch + j]>max)
        max = Z[L-1][i * nbatch + j];
        guess = i;}
  if(guess == test_label[j + iter*nbatch])
      hits+=1;
}

}
printf("success rate:%f\n",(double)hits/total);
}
