#define e 2.71828
#include "mnist.h"
#include "omp.h"
#include<time.h>
#include "math.h"
#include <stdio.h>
#include"/opt/homebrew/Cellar/openblas/0.3.26/include/cblas.h"

void shuffle(int* arr, int n) {
  srand(time(NULL)); // Seed the random number generator
  for (int i = n - 1; i > 0; i--) {
    int j = rand() % (i + 1); // Generate random index within 0 to i
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
  }
}

double relu(double z)
{
    return (z > 0) ? (z):(0);
}

int*lable(int l)
{
  int *lab =  (int*) calloc(OUT_NUM, sizeof(int));
  lab[l] = 1;
  return lab;
}


double relu_prime(double z)
{
    if (z>0)
        return 1.0;
    return 0.0;
}


void hand_mm(double * A, double * B, double * C, int m, int k, int n)
{
  int cur_row=0, cur_col=0;
  for(int i=0;i < m * n;i++)
    {
     cur_row = i/n;
     cur_col = i - cur_row * n;
     for(int j=0;j<k;j++)
           C[i] += A[cur_row * k + j] * B[cur_col + j * n];
    } 
}

void initialize(double* W[], double*B[],int L, int nhidden)
{
W[0]=NULL;
B[0]=NULL;
W[1] = (double*) malloc(sizeof(double) * nhidden * SIZE);
B[1] = (double*) malloc(sizeof(double) * nhidden);

double stddev = sqrt(2.0/SIZE);
for(int i=0;i<SIZE*nhidden;i++)
  W[1][i] = stddev * ((double) rand() / (RAND_MAX / 2.0f) - 1.0f);

stddev = sqrt(2.0/nhidden);
for(int i=0;i<nhidden;i++)
  B[1][i] = stddev * ((double) rand() / (RAND_MAX / 2.0f) - 1.0f);

for(int i=2;i <L-1 ;i++)
{
  W[i] = (double*) malloc(sizeof(double) * nhidden*nhidden);
  for(int j=0;j<nhidden*nhidden;j++)
      W[i][j] = stddev * ((double) rand() / (RAND_MAX / 2.0f) - 1.0f);
  
  B[i] = (double*) malloc(sizeof(double) * nhidden);
  for(int j=0;j<nhidden;j++)
      B[i][j] = stddev * ((double) rand() / (RAND_MAX / 2.0f) - 1.0f);

}

W[L-1] = (double*) malloc(sizeof(double) * OUT_NUM*nhidden);
  for(int j=0;j<OUT_NUM*nhidden;j++)
      W[L-1][j] = stddev * ((double) rand() / (RAND_MAX / 2.0f) - 1.0f);

B[L-1] = (double*) malloc(sizeof(double) * OUT_NUM);
stddev = sqrt(2.0/OUT_NUM);
for(int j=0;j<OUT_NUM;j++)
    B[L-1][j] = stddev * ((double) rand() / (RAND_MAX / 2.0f) - 1.0f);
}

void initialize_u(double* uW[], double*uB[],int L, int nhidden)
{
uW[0]=NULL;
uB[0]=NULL;
uW[1] = (double*) calloc(nhidden * SIZE, sizeof(double));
uB[1] = (double*) calloc(nhidden, sizeof(double));
for(int i=2;i <L-1 ;i++)
{
  uW[i] = (double*) calloc(nhidden*nhidden, sizeof(double));  
  uB[i] = (double*) calloc(nhidden,sizeof(double));
}

uW[L-1] = (double*) calloc(OUT_NUM*nhidden, sizeof(double));
uB[L-1] = (double*) calloc(OUT_NUM, sizeof(double));
}

void unset(double*uW[], double*uB[], int L, int nhidden)
{
for(int i=0;i<nhidden*SIZE;i++)
  uW[1][i] = 0;
for(int i=0;i<nhidden;i++)
  uB[1][i] = 0;

for(int l=2;l <L-1 ;l++)
{
  for(int i=0;i<nhidden*nhidden;i++)
    uW[l][i] = 0;  
  for(int i=0;i<nhidden;i++)  
    uB[l][i] =0;
}

for(int i=0;i<OUT_NUM*nhidden;i++)
    uW[L-1][i] = 0;
for(int i=0;i<OUT_NUM;i++)
    uB[L-1][i] = 0;

}

double* trans(double * M, int row, int col)
{
    double * MT = (double*) malloc(sizeof(double) * row *col);
    for(int i=0;i<row;i++)
      for(int j=0;j<col;j++)
        MT[j*row+i]= M[i*col+j];
    return MT;
}

int
//main(int argc, char** argv)
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
int nhidden = 800, nepoch = 5, nbatch = 256;

double alpha = 0.01;
int L = nlayer + 2;
int *y = (int*)malloc(sizeof(int) * OUT_NUM * nbatch);
//parameters
//W[0] and b[0] are not used
double* W[L];
double* B[L];
initialize(W, B, L, nhidden);
double sum=0,t=0,debug=0;


//below arrays are used to store update values of W and B
double*uW[L];
double* uB[L];
double*uW_acc[L];
double* uB_acc[L];
initialize_u(uW, uB,L,nhidden);
initialize_u(uW_acc, uB_acc,L, nhidden);
//activatin matrix
double * a[L];
a[0] =(double*) malloc(SIZE*nbatch);
for(int i=1;i<L-1;i++)
  a[i] = (double*) malloc(nhidden*nbatch);

a[L-1]= (double*) malloc(OUT_NUM*nbatch);

//pre action matrix Z[0] is not used
double * Z[L];
Z[0]=NULL;
for(int i=1;i<L-1;i++)
  Z[i] = (double*) malloc(nbatch*nhidden*sizeof(double));

Z[L-1]= (double*) malloc(nbatch*OUT_NUM*sizeof(double));

//errors for hidden layers
//delta[0] is not used
double *delta[L];
delta[0]=NULL;
for(int i=1;i<L-1;i++)
  delta[i] = (double*)malloc(nbatch*nhidden*sizeof(double));

//errors for output layer
delta[L-1] = (double*)malloc(nbatch*OUT_NUM*sizeof(double));

//array used to support shuffle training data
int *rand_num=(int*)malloc(sizeof(int)*NUM_TRAIN);
for(int i=0;i<NUM_TRAIN;i++)
    rand_num[i]=i;

double *Ta; 
double * WT;
double t0 = omp_get_wtime();
//start to forward_propagation
int niter = NUM_TRAIN/nbatch;
for(int epo =0; epo<nepoch-2;epo++)
{
//shuffle tranining set for later use.
shuffle(rand_num, NUM_TRAIN);
for(int iter=0;iter<niter;iter++)
{   
    printf("training on epoch %d batch%d\n", epo, iter);
    debug=0;
    for(int i=0;i<nhidden*nhidden;i++)
      debug+=W[1][i];
    //printf("W[1] sum %f\n", debug);
    
    //set update matrix values to 0
    unset(uW_acc, uB_acc, L, nhidden);
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
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                nhidden, nbatch, SIZE, 1.0, W[1], SIZE, a[0], 
                nhidden, 0.0, Z[1], nbatch);
      //printf("%s \n", "Z[1]");
      for(int i=0;i<nhidden;i++)
        //printf("%f ", Z[1][i]);
        putchar('\n');
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
        WT = trans(W[L-1],OUT_NUM,nhidden);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        nhidden,nbatch, OUT_NUM, 1.0, WT,OUT_NUM, delta[L-1], nbatch
        ,0.0,delta[L-2],nbatch);
        free(WT);
        printf("%s","delta1:\n");
        for(int i=0;i<10;i++)
            printf("%f ", delta[1][i]);
        putchar('\n');
        for(int i=0;i<nhidden*nbatch;i++)
           delta[L-2][i] = delta[L-2][i] * relu_prime(Z[L-2][i]);
        printf("%s","delta1:\n");
        for(int i=0;i<10;i++)
            printf("%f ", delta[1][i]);
        putchar('\n');
        //compute following hidden layer error
        for(int l=L-3;l>=1;l--)
        {
            WT = trans(W[l+1],nhidden,nhidden);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            nhidden,nbatch, nhidden, 1, WT,nhidden, delta[l+1], nbatch
            ,0.0,delta[l],nbatch);
            for(int i=0;i<nhidden*nbatch;i++)
                delta[L-2][i] = delta[L-2][i] * relu_prime(Z[L-2][i]);
            free(WT);
        }
        
        //compute values used to update W and B
        //last layer

        Ta = trans(a[L-2], nhidden, nbatch);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            OUT_NUM,nbatch,nhidden, 1, delta[L-1], nhidden, Ta, nhidden
            ,0.0,uW[L-1],nhidden);
        for(int i=0;i<OUT_NUM*nhidden;i++)
          uW_acc[L-1][i] += uW[L-1][i];
        for(int i=0;i<OUT_NUM;i++)
          uB_acc[L-1][i] += delta[L-1][i];
        free(Ta);
        for(int l=L-2;l>1;l--)
        {
        Ta = trans(a[l-1],nhidden, nbatch);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            nhidden,nhidden,nbatch, 1.0, delta[l], nbatch, Ta, nhidden
            ,0.0,uW[l],nhidden);
        for(int i=0;i<nhidden*nhidden;i++)
          uW_acc[l][i] += uW[l][i];
        for(int i=0;i<nhidden;i++)
          uB_acc[l][i] += delta[l][i];
        free(Ta);
        }

        //computer layer 1

        Ta = trans(a[0],SIZE, nbatch);
        printf("%s", "Ta[0]:\n");
        for(int i=100;i<200;i++)
          printf("%f ", Ta[i]);
          putchar('\n');
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            nhidden,SIZE,nbatch, 1.0, delta[1], nbatch, Ta, SIZE
            ,0.0,uW[1],SIZE);
        free(Ta);
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
double t1 = omp_get_wtime();



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
double t2 = omp_get_wtime();
printf("time used to train%f\n",t1-t0);
printf("time used to inference%f\n",t2-t1);
printf("success rate:%f\n",(double)hits/total);

}//main end


    



