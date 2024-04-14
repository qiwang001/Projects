#define e 2.71828
#include "mnist.h"
#include "omp.h"
double sigmoid(double x)
{
return 1/(1 + pow(e, -x));
}

int*lable(int l)
{
  int *lab =  (int*) calloc(OUT_NUM, sizeof(int));
  lab[l] = 1;
  return lab;
}

void shuffle(int* arr, int n) {
  srand(time(NULL)); // Seed the random number generator
  for (int i = n - 1; i > 0; i--) {
    int j = rand() % (i + 1); // Generate random index within 0 to i
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
  }
}

double sigmoid_prime(double z)
{
    double a = sigmoid(z);
    return a * (1 - a);
}

int
//main(int argc, char** argv)
main()
{
/*int nlayer = atoi(argv[1]), nhidden = atoi(argv[2]), nepoch = atoi(argv[3]), nbatch = atoi(argv[4]);
double alpha = atof(argv[5]);
*/
int nlayer = 2;
unsigned int const nhidden = 10, nepoch = 5, nbatch = 1000;
double alpha = 0.01;
int L = nlayer + 2;
double w_input[nhidden][SIZE], b_input[nhidden], w_hidden[nlayer-1][nhidden][nhidden],
b_hidden[nlayer-1][nhidden], w_output[OUT_NUM][nhidden], b_output[OUT_NUM],sum=0;
int i,j,k,l,b,sample=10,p,*lab=NULL;
double Z[nbatch][L-1][nhidden], a[nbatch][L-1][nhidden];
double delta[nbatch][L-1][nhidden];
double temp[nhidden][nhidden];
double temp1[nhidden];
double temp2[OUT_NUM][nhidden];
double temp3[OUT_NUM];
double temp4[nhidden][SIZE];
double temp5[nhidden];
int batch_ite = NUM_TRAIN/nbatch;
int *rand_num=(int*)malloc(sizeof(int)*NUM_TRAIN);

for(int i=0;i<NUM_TRAIN;i++)
    rand_num[i]=i;

for(i=0;i<nhidden;i++)
    {
    for(j=0;j<nhidden;j++)
        temp[i][j]=0;
    temp1[i]=0;
    }

for(i=0;i<OUT_NUM;i++)
{ 
    for(j=0;j<nhidden;j++)
        temp2[i][j]=0;
    temp3[i]=0;
}

for(i=0;i<nhidden;i++)
{ 
    for(j=0;j<SIZE;j++)
        temp4[i][j]=0;
    temp5[i]=0;
}

srand(time(NULL));
load_mnist();
//initialize w_input and b_input
for(int i=0;i<nhidden;i++)
{    for(int j=0;j<SIZE;j++)
        w_input[i][j] = (double)rand()/(RAND_MAX + 1.0);
    b_input[i] = (double)rand()/(RAND_MAX + 1.0);
}
printf("%s\n", "w_input: ");
for(int i=0;i<nhidden;i++)
{
    for(int j=0;j<5;j++)
        printf("%f ", w_input[i][j]);
    putchar('\n');
}    

printf("%s\n", "b_input:");
for(int i=0;i<nhidden;i++)
        printf("%f\n", b_input[i]);

// initialize w_hidden and b_hidden
for(k = 0;k < nlayer-1;k++)
    for(i = 0;i < nhidden; i++)
       {
        for(j=0;j < nhidden; j++)
            w_hidden[k][i][j] = (double)rand()/(RAND_MAX + 1.0);
        b_hidden[k][i] = (double)rand()/(RAND_MAX + 1.0);
       }

printf("%s\n", "w_hidden: ");
for(k = 0;k < nlayer-1;k++)
    {
    for(i = 0;i < nhidden; i++)
        {
            for(j=0;j < nhidden; j++)
                printf("%f ",w_hidden[k][i][j]);
            putchar('\n');
        }
    putchar('\n');
    }
    
printf("%s\n", "b_hidden: ");
for(k = 0;k < nlayer-1;k++)
    {
    for(i = 0;i < nhidden; i++)
        printf("%f\n",b_hidden[k][i]);
    putchar('\n');
    }

// initialize w_out and b_out
for(i=0;i<OUT_NUM;i++)
    {
    for(j=0;j < nhidden;j++)
        w_output[i][j] = (double)rand()/(RAND_MAX + 1.0);
    b_output[i] = (double)rand()/(RAND_MAX + 1.0);
    }

printf("%s\n", "w_output: ");
for(i=0;i<OUT_NUM;i++)
    {
    for(j=0;j < nhidden;j++)
        printf("%f ",w_output[i][j]);
    putchar('\n');
    }

printf("\n%s\n", "b_output: ");
for(i=0;i<OUT_NUM;i++)
        printf("%f\n",b_output[i]);

double t1 = omp_get_wtime();
//start to train 
for(p = 0;p < nepoch; p++)
{
shuffle(rand_num, NUM_TRAIN);
for(int iter=0;iter<batch_ite;iter++)
{
for(int b=0;b < nbatch ; b++)
{
    if(0)
    {
        printf("b=%d\n, w_hidden:\n",b);
        for(k = 0;k < nlayer-1;k++)
        {
        for(i = 0;i < nhidden; i++)
            {
                for(j=0;j < nhidden; j++)
                    printf("%f ",w_hidden[k][i][j]);
                putchar('\n');
            }
        putchar('\n');
    }
    }
    sample = rand_num[iter * nbatch + b];
    // compute the first hidden layer's activation values
    for(i = 0;i < nhidden; i++)
        {
        sum= 0;
        for(int j = 0;j < SIZE; j++)
            sum += train_image[sample][j] * w_input[i][j];
        Z[b][0][i] = sum + b_input[i];
        a[b][0][i] = sigmoid(Z[b][0][i]);
        }

    // compute the following hidden layer's activation values
    for (l = 1; l < L-2; l++)
        for(i=0; i < nhidden; i++)
            {   
                sum = 0;
                for(j = 0;j < nhidden; j++)
                    sum += w_hidden[l-1][i][j] * a[b][l-1][j];  
                Z[b][l][i] = sum + b_hidden[l-1][i];       
                a[b][l][i] = sigmoid(Z[b][l][i]); 
            }

    //compute the output layer's  values
    for(i = 0; i < OUT_NUM; i++)
        {
        sum = 0;
        for(j = 0;j < nhidden; j++)
            sum += w_output[i][j] * a[b][L-3][j];
        Z[b][L-2][i] = sum + b_output[i];
        a[b][L-2][i] = sigmoid(Z[b][L-2][i]);
        }

    //computer error in last/output layer
    lab = lable(train_label[sample]);
    for(i = 0;i < OUT_NUM; i++)
        delta[b][L-2][i] = (a[b][L-2][i] - lab[i]) * a[b][L-2][i] * (1-a[b][L-2][i]);

    //backpropagate 
    // first compute the (L-2)th layer
    // using equation transpose(w)[k][j] = w[j][k]
    for(j=0;j < nhidden; j++)
        for(k = 0;k < OUT_NUM; k++)
            delta[b][L-3][j] = delta[b][L-2][k] * w_output[k][j]* a[b][L-3][i] * (1-a[b][L-3][i]);

    //compute (L-4)th layer to the 2nd layer, which is the first hidden layer
    for(l = L-4; l >= 0;l--)
        for(j=0; j < nhidden;j++)
            for(k=0; k < nhidden; k++)
                delta[b][l][k] = delta[b][l+1][k] * w_hidden[l+1][k][j] * a[b][l][i] * (1-a[b][l][i]);
}//one batch end
/*printf ("after %d set of examples, activation value:\n", nbatch);
for(int b=0;b<nbatch;b++)
    {for(int l=0;l<L-1;l++)
        {
            for(int i=0;i<nhidden;i++)
                printf("%f ", a[b][l][i]);
            putchar('\n');
        }
    putchar('\n');
    }
*/
for(l=0;l<L-2;l++)
{
//calculate the sum of error in a batch for hidden layer
for(i=0;i<nhidden;i++)
    for(j=0;j<nhidden;j++)
        for(int b=0;b < nbatch;b++)
            temp[i][j] += delta[b][l][i] * a[b][l][j];
            
for(i=0; i<nhidden; i++)
    for(int b=0;b < nbatch;b++)
        temp1[i] += delta[b][l][i];
/*
printf("before updating: temp is :\n");
for(i=0;i < nhidden;i++)
    {
    for(j=0;j < nhidden;j++)
        printf("%f ",temp[i][j]);
    putchar('\n');
    }
*/

//update hidden layer's parameters
for(i=0;i < nhidden;i++)
    {
    for(j=0;j < nhidden;j++)
        w_hidden[l][i][j] = w_hidden[l][i][j] - (double)(alpha * temp[i][j]/nbatch);
    b_hidden[l][i] = b_hidden[l][i] - (double)(alpha * temp1[i]/nbatch);
    }

//reset temp variables
for(i=0;i < nhidden;i++)
    for(j=0;j < nhidden;j++)
        temp[i][j]=0;

for(i=0;i < nhidden;i++)
    temp1[i] =0;;
}

//calculate the sum of error in a batch for output layer
for(i=0; i < OUT_NUM;i++)
    for(j=0;j < nhidden;j++)
        for(b=0; b < nbatch;b++)
            temp2[i][j] += delta[b][L-2][i] * a[b][L-3][j];

for(i=0;i < nhidden; i++)
    for(int b=0;b < nbatch; b++)
        temp3[i] += delta[b][L-2][i];

//update output layer's parameters
for(i=0;i < OUT_NUM; i++)
{
    for(j=0;j < nhidden;j++)
        w_output[i][j] = w_output[i][j] - ((double)alpha/nbatch) * temp2[i][j];
    b_output[i] = b_output[i] - ((double)alpha/nbatch) * temp3[i];
}


//calculate the error for the first hidden layer

for(i=0; i<nhidden; i++)
    for(j=0;j<SIZE; j++)
        for(b=0; b<nbatch;b++)
            temp4[i][j] += delta[b][0][i] * train_image[sample][j];


for(i=0;i < nhidden; i++)
    for(int b=0;b < nbatch; b++)
        temp5[i] += delta[b][L-2][i];


//update the input parameters
for(i=0;i < nhidden; i++)
{
    for(j=0;j < SIZE;j++)
        w_input[i][j] = w_input[i][j] - ((double)alpha/nbatch) * temp4[i][j];
    b_input[i] = b_input[i] - ((double)alpha/nbatch) * temp5[i];
}

//reset temp variables
for(i=0;i<nhidden;i++)
{ 
    for(j=0;j<SIZE;j++)
        temp4[i][j]=0;
    temp5[i]=0;
}

}//batch iter end

}//epoch end
printf("after training %d epoch:\n w_hidden:\n",nepoch);
for(i=0;i<nhidden;i++)
    {for(j=0;j<nhidden;j++)
        printf("%f ", w_hidden[0][i][j]);
        putchar('\n');
        }

printf("%s", "b_hidden:\n");
for(i=0;i<nhidden;i++)
    {
        printf("%f ", b_hidden[0][i]);
        putchar('\n');
    }

printf("%s", "w_output:\n");
for(i=0;i<OUT_NUM;i++)
    {for(j=0;j<nhidden;j++)
        printf("%f ", w_output[i][j]);
    putchar('\n');}

printf("%s", "b_output:\n");
for(i=0;i<OUT_NUM;i++)
    {
        printf("%f ", b_output[i]);
        putchar('\n');
    }
putchar('\n');
double t2 = omp_get_wtime();
//run NN on test images
int hit = 0;
int total=1000;
double max=0;
int guess=0;
for(int b=0;b<total;b++)
{
    // compute the first hidden layer's activation values
    for(i = 0;i < nhidden; i++)
        {
        sum= 0;
        for(int j = 0;j < SIZE; j++)
            sum += train_image[b][j] * w_input[i][j];
        Z[b][0][i] = sum + b_input[i];
        a[b][0][i] = sigmoid(Z[b][0][i]);
        }

    // compute the following hidden layer's activation values
    for (l = 1; l < L-2; l++)
        for(i=0; i < nhidden; i++)
            {   
                sum = 0;
                for(j = 0;j < nhidden; j++)
                    sum += w_hidden[l-1][i][j] * a[b][l-1][i];  
                Z[b][l][i] = sum + b_hidden[l-1][i];       
                a[b][l][i] = sigmoid(Z[b][l][i]); 
            }

    //compute the output layer's  values
    for(i = 0; i < OUT_NUM; i++)
        {
        sum = 0;
        for(j = 0;j < nhidden; j++)
            sum += w_output[i][j] * a[b][L-3][j];
        Z[b][L-2][i] = sum + b_output[i];
        a[b][L-2][i] = sigmoid(Z[b][L-2][i]);
        }
    printf("test image%d's output:\n",b);
    putchar('\n');
    for(int i=0;i<OUT_NUM;i++)
        printf("%f\n", a[b][L-2][i]);
    
     max = a[b][L-2][0];
     guess = 0;
    for(int i=0; i < OUT_NUM; i++)
        {
        if(a[b][L-2][i] > max)
            {
            max = a[b][L-2][i];
            guess = i;
            }
        }
    printf("I guess it's:%d\n" ,guess);
    if(guess == test_label[b])
        hit += 1;
    printf("In fact it's:%d\n" ,test_label[b]);
}

double t3 = omp_get_wtime();
double success_rate = (double)hit/total;
printf("hit: %d, total: %d,success_rate:%1.2f\n", hit, total, success_rate);
printf("Time used to train 60K images:%f\n", t2-t1);
printf("Time to infer the 10K images:%f\n", t3-t2);
}//main end