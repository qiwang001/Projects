#include "mnist.h"
#define e 2.71828
double sigmoid(double x)
{
return 1/(1 + pow(e, -x));
}
double * forward(int L, int nhidden,int sample,double **w_input,double* b_input,double*** w_hidden, double** b_hidden, double **w_output, double*b_output,double ***Z,double ***a,int b)
{
double sum=0;
//compute the first hidden layer's activation values
    for(int i = 0;i < nhidden; i++)
        {
        sum= 0;
        for(int j = 0;j < SIZE; j++)
            sum += train_image[sample][j] * w_input[i][j];
        Z[b][1][i] = sum + b_input[i];
        a[b][1][i] = sigmoid(Z[b][1][i]);
        }

    // compute the following hidden layer's activation values
    for (int l = 2; l < L-1; l++)
        for(int i=0; i < nhidden; i++)
            {   
                sum = 0;
                for(int j = 0;j < nhidden; j++)
                    sum += w_hidden[l][i][j] * a[b][l-1][i];  
                Z[b][l][i] = sum + b_hidden[l][i];       
                a[b][l][i] = sigmoid(Z[b][l][i]); 
            }
    //compute the output layer's  values
    for(int i = 0; i < OUT_NUM; i++)
        {
        sum = 0;
        for(int j = 0;j < nhidden; j++)
            sum += w_output[i][j] * a[b][L-2][j];
        Z[b][L-1][i] = sum + b_output[i];
        a[b][L-1][i] = sigmoid(Z[b][L-1][i]);
        }
    

}