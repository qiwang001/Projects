#define e 2.71828
#include "mnist.h"
#include "omp.h"
#include<time.h>
#include "math.h"

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
        return 1;
    return 0;
}


double** matrix_maker(int row, int col)
{
    double* g = (double *)malloc(row * col * sizeof(double));
    double ** M = (double**)malloc(sizeof(double*) * row);
    for(int i=0;i<row;i++)
        M[i] = g + i * col;

    //kaiming initialization
    double stddev = sqrt(2.0 / col);
    for(int i=0;i<row;i++)
        for(int j=0;j<col;j++)
            M[i][j] = stddev * ((double) rand() / (RAND_MAX / 2.0f) - 1.0f);
    return M;
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
int*y = NULL;

//parameters
double**W[L-1];
double* B[L-1];
W[0] = matrix_maker(nhidden, SIZE),
B[0] = (double*)malloc(sizeof(double) * nhidden);
double stddev = sqrt(2.0 / nhidden);
for(int j =0;j<nhidden;j++)
    B[0][j] = stddev * ((double) rand() / (RAND_MAX / 2.0f) - 1.0f);

for(int i=1;i<L-2;i++)
{
    W[i] = matrix_maker(nhidden, nhidden);
    B[i] = (double*)malloc(sizeof(double) * nhidden);
    for(int j =0;j<nhidden;j++)
        B[i][j] = stddev * ((double) rand() / (RAND_MAX / 2.0f) - 1.0f);
}

W[L-2]=matrix_maker(OUT_NUM, nhidden);
B[L-2]= (double*)malloc(sizeof(double) * OUT_NUM);
double stddev_out = sqrt(2.0 / OUT_NUM);
for(int j =0;j<OUT_NUM;j++)
    B[L-2][j] = stddev_out * ((double) rand() / (RAND_MAX / 2.0f) - 1.0f);

double sum=0,t=0,debug=0;

//below arrays are used to store update values of W and B
double**uW[L-1];
double* uB[L-1];
uW[0] = matrix_maker(nhidden, SIZE),
uB[0] = (double*)malloc(nhidden*sizeof(double));

for(int i=1;i<L-2;i++)
{
    uW[i] = matrix_maker(nhidden, nhidden);
    uB[i] = (double*)malloc(sizeof(double) * nhidden);
}

uW[L-2]=matrix_maker(OUT_NUM, nhidden);
uB[L-2]= (double*)malloc(sizeof(double) * OUT_NUM);

//activations, a[0][] is input layer
double*temp_a = (double*) malloc(sizeof(double) * (SIZE + nhidden*(L-2) + OUT_NUM));
double**a = (double**) malloc(sizeof(double*)*(L-1));
a[0] = temp_a;
a[1] = temp_a + SIZE;
for(int i=2; i<=L-1 ;i++)
    a[i] = a[1] + (i-1) * nhidden;


double*temp_Z = (double*) malloc(sizeof(double) * (SIZE + nhidden*(L-2) + OUT_NUM));
double**Z = (double**) malloc(sizeof(double*)*(L-1));
Z[0] = temp_Z;
Z[1] = temp_Z + SIZE;
for(int i=2; i<=L-1 ;i++)
    Z[i] = Z[1] + (i-1) * nhidden;

//errors for hidden layers
//double delta[L-2][nhidden];
double * g = (double*) calloc((nhidden* (L-2) + OUT_NUM), sizeof(double));
double ** delta = (double**) malloc(sizeof(double*) * (L-1));
for(int l=0;l<= L-2;l++)
    delta[l] = g + l * nhidden;


//array used to support shuffle training data
int *rand_num=(int*)malloc(sizeof(int)*NUM_TRAIN);
for(int i=0;i<NUM_TRAIN;i++)
    rand_num[i]=i;

int sample=0;
double t0 = omp_get_wtime();
//start to forward_propagation
int batch_ite = NUM_TRAIN/nbatch + 1;
for(int epo =0; epo<nepoch;epo++)
{
//shuffle tranining set for later use.
shuffle(rand_num, NUM_TRAIN);

for(int iter=0;iter<batch_ite;iter++)
{   
    printf("training on epoch %d batch%d\n", epo, iter);
    for(int i=0;i<nhidden;i++)
        {for(int j=0;j< SIZE;j++)
           uW[0][i][j]=0;
        uB[0][i]=0;}
    
    for(int l= 1;l<L-2;l++)
        for(int i=0;i<nhidden;i++)
            {
            for(int j=0;j<nhidden;j++)
                uW[l][i][j]=0;
            uB[l][i] = 0;
            }

    for(int i =0;i<OUT_NUM;i++)
        {for(int j=0;j<nhidden;j++)
            uW[L-2][i][j] = 0;
        uB[L-2][i]=0;
        } 

    //start training on a batch
    for(int b=0;b < nbatch ; b++){    
        //randomly selected index of training image
        sample = rand_num[iter * nbatch + b];
        //compute a[0][]
        for(int i=0;i<SIZE;i++)
            a[0][i] = train_image[sample][i];

        //forward propagation 
        for(int l=1;l<=L-2;l++)
            {
            for(int i=0;i<nhidden;i++)
                {   sum = 0;
                    for(int j=0;j<((l)>(1)?(nhidden):(SIZE));j++)
                        sum +=  W[l-1][i][j] * a[l-1][j];
                    Z[l][i] = sum + B[l-1][i];
                    a[l][i] = relu(Z[l][i]);
                }
            }        
        //for(int i=0;i<OUT_NUM;i++)
          //  printf("%f \n", Z[2][i]);

        //compute the output layer using softmax activation
          double sum1= 0;
          for(int i=0;i<OUT_NUM;i++)
            {sum = 0;
            for(int j=0;j<nhidden;j++)
                sum += W[L-2][i][j] * a[L-2][j];
            Z[L-1][i] = sum + B[L-2][i];
            sum1 += pow(e, Z[L-1][i]);
            }
        
        for(int i=0;i<OUT_NUM;i++)
            a[L-1][i] = pow(e, Z[L-1][i])/sum1;

        //Backpropagate 
        //compute output layer error
        //using cross entrophy loss
        y = lable(train_label[sample]);
        for(int i = 0;i < OUT_NUM; i++)
            delta[L-2][i] = (a[L-1][i] - y[i]);
            
        //compute hidden layer errors 
        for(int l=L-3; l>=0; l--)
            for(int i=0;i<nhidden;i++)
                for(int j=0;j<((l==L-3)?(OUT_NUM):(nhidden)); j++)
                    delta[l][i] = W[l+1][j][i] * delta[l+1][j] * relu_prime(Z[l+1][i]);
        
        //compute values used to update W and B
        for(int i=0;i < nhidden;i++)
            {for(int j=0;j < SIZE;j++)
                uW[0][i][j] += delta[0][i] * a[0][j];
            uB[0][i] += delta[0][i];
            }

        //compute following layer update values
        for(int l=1;l< L-2 ;l++)
            for(int i=0;i<nhidden;i++)
            {   for(int j=0;j<nhidden;j++)
                    uW[l][i][j] += delta[l][i]*a[l][j];
                uB[l][i] += delta[l][i]; 
            }
    
        //compute the output layer's update values
        for(int i=0;i<OUT_NUM;i++)
            {for(int j=0;j<nhidden;j++)
                uW[L-2][i][j] += delta[L-2][i] *a[L-2][j];
            uB[L-2][i] += delta[L-2][i];  
            }
        }// one batch end
/*
    printf("start gradient decent on batch%d\nuW[0]:\n", iter);
    for(int i=100;i<110;i++)
    {    for(int j=100;j<120;j++)
            printf("%1.3f ",uW[0][i][j]);
        putchar('\n');
    }
        printf("start gradient decent on batch%d\nuW[1]:\n", iter);
    for(int i=0;i<OUT_NUM;i++)
    {    for(int j=100;j<120;j++)
            printf("%1.3f ",uW[1][i][j]);
        putchar('\n');
    }*/



    //gradient decent
    for(int i=0;i<nhidden;i++)
    {    for(int j=0;j<SIZE;j++)
            W[0][i][j] = W[0][i][j] - (double)(alpha/nbatch)*uW[0][i][j];
         B[0][i] = B[0][i] - (double)(alpha/nbatch)*uB[0][i];
    }

    for(int l=1;l<L-2;l++)
        for(int i=0;i<nhidden;i++)
        {  for(int j=0;j<nhidden;j++)
                W[l][i][j] = W[l][i][j] - (double)(alpha/nbatch)*uW[l][i][j];
            B[l][i] = B[l][i] - (double)(alpha/nbatch)*uB[l][i];
        }

    for(int i=0;i<OUT_NUM;i++)
        {for(int j=0;j<nhidden;j++)
            W[L-2][i][j] = W[L-2][i][j] - (double)(alpha/nbatch)*uW[L-2][i][j];
        B[L-2][i]=B[L-2][i] - (double)(alpha/nbatch)*uB[L-2][i];
        }
        
    /*for(int i=0;i<OUT_NUM;i++)
        {for(int j=0;j<10;j++)
            printf("%1.4f ", (double)(alpha/nbatch)*uW[1][i][j]);
        putchar('\n');}

    for(int i=0;i<SIZE;i++)
        {for(int j=100;j<200;j++)
            printf("%1.8f ", (double)(alpha/nbatch)*uW[0][i][j]);
        putchar('\n');}*/

    debug=0;
    for(int i=0;i<OUT_NUM;i++)
        for(int j=0;j<nhidden;j++)
            debug += W[1][i][j];
    debug = debug /(OUT_NUM*nhidden);
    printf("W[1] mean %f\n",debug);
    printf("W[1][5][100] %f\n",W[1][5][100]);
    printf("W[1][5][200] %f\n",W[1][5][200]);
    printf("B[1][5] %f\n",B[1][5]);
    printf("B[0][200] %f\n",B[0][200]);
    debug=0;
    for(int i=0;i<nhidden;i++)
        for(int j=0;j<SIZE;j++)
            debug += W[0][i][j];
    
    debug = debug /(nhidden*SIZE);
    printf("W[0] mean %f\n",debug);

    /*
    printf("%s","W[0]:\n");
    for(int i=0;i<30;i++)
        {for(int j=0;j<30;j++)
            printf("%1.3f ", W[0][i][j]);
         putchar('\n');}*/
}// one epoch end

}// traininig end
double t1 = omp_get_wtime();
// start testing
int hits=0, guess=0;
double max=-100000;
for(int sample=0;sample<NUM_TEST;sample++)
{
for(int i=0;i<SIZE;i++)
    a[0][i] = test_image[sample][i];
    //forward propagation 
    for(int l=1;l<=L-2;l++)
        {
        for(int i=0;i<nhidden;i++)
            {   sum = 0;
                for(int j=0;j<((l)>(1)?(nhidden):(SIZE));j++)
                    sum +=  W[l-1][i][j] * a[l-1][j];
                Z[l][i] = sum + B[l-1][i];
                a[l][i] = relu(Z[l][i]);
            }
        }        

    //compute the output layer using softmax activation
        t=0;
        for(int i=0;i<OUT_NUM;i++)
        {sum = 0;
        for(int j=0;j<nhidden;j++)
            sum += W[L-2][i][j] * a[L-2][j];
        Z[L-1][i] = sum + B[L-2][i];
        t += pow(e, Z[L-1][i]);
        }

    for(int i=0;i<OUT_NUM;i++)
        {
        a[L-1][i] = pow(e, Z[L-1][i])/t;
        if (a[L-1][i] > max)
            {max = a[L-1][i];
            guess = i;}
        }
    
    if(guess==test_label[sample])
        hits+=1;

}
double t2 = omp_get_wtime();
printf("time used to train%f\n",t1-t0);
printf("time used to inference%f\n",t2-t1);
printf("success rate:%f\n",(double)hits/NUM_TEST);

}