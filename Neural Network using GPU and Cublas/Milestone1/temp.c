
double *twi = (double*)malloc(sizeof(double) * SIZE * nhidden);
double **w_input = (double**) malloc(sizeof(double*) * nhidden);
for(i = 0;i < nhidden; i++)
    w_input[i] = twi + i * SIZE; 


double *tw = (double*)malloc(sizeof(double) * (nlayer-1) * nhidden * nhidden);
double **ttw = (double**)malloc(sizeof(double**) * (nlayer-1) * nhidden);
double ***w_hidden= (double***)malloc(sizeof(double***) * (nlayer - 1));

for(i = 0;i < nhidden * (nlayer - 1); i++)
    ttw[i] = tw + nhidden * i; 

for(i = 0;i < (nlayer - 1); i++)
    w_hidden[i] = ttw + nhidden * i; 





double *tb = (double*)malloc(sizeof(double) * (nlayer-1) * nhidden);
double **b_hidden= (double**) malloc(sizeof(double*) * (nlayer - 1));
for(i = 0;i < nlayer - 1; i++)
    b_hidden[i] = tb + nhidden * i; 



double *two = (double*)malloc(sizeof(double) * OUT_NUM * nhidden);
double **w_output = (double**) malloc(sizeof(double*) * nhidden);
for(i = 0; i<OUT_NUM;i++)
    w_output[i] = two + i * nhidden;

// initialize w and b using formalized random numbers.
for(int i = 0;i < nlayer - 1;i++)
    {   
    for(int j = 0;j < nhidden;j++)
        {
            w_hidden[i][j] = (double)rand()/(RAND_MAX + 1.0);
            b_hidden[i][j] = (double)rand()/(RAND_MAX + 1.0);
        }
    }

for(int i = 0;i < nhidden;i++)
    {
    for(int j=0;j<SIZE;j++)
        w_input[i][j] = (double)rand()/(RAND_MAX + 1.0);
    b_input[i] = (double)rand()/(RAND_MAX + 1.0);
    }
    
for(int i = 0;i < OUT_NUM; i++)
    {
        for(int j=0; j < nhidden; j++)
            w_output[i][j] = (double)rand()/(RAND_MAX + 1.0);
        b_output[i] = (double)rand()/(RAND_MAX + 1.0);
    }


double* feed_forward(int nlayer, int nhidden, double** w_hidden, double** b_hidden, double* b_input, double **w_input, double ** w_output, double* b_output, int sample)
{
double  *h_pre = (double*) malloc(sizeof(double) * nhidden),
        *h = (double*) malloc(sizeof(double) * nhidden),
        *out = (double*)malloc(sizeof(double) * OUT_NUM),
        sum;
int i,j;

// compute the first hidden layer's activation values
for(i = 0;i < nhidden; i++)
    {
    sum= 0;
    for(int j = 0;j < SIZE; j++)
        sum += train_image[sample][j] * w_input[i][j];
    h_pre[i] =  sigmoid(sum + b_input[i]);
    }

// compute the following hidden layer's activation values
for (int k=0; k < nlayer-1; k++)
{
    for(i=0; i < nhidden; i++)
        {
            sum = 0;
            for(j = 0;j < nhidden; j++)
                sum += w_hidden[k][j] * h_pre[j];            
            h[i] = sigmoid(sum + b_hidden[k][i]);            
        }
    for(i=0;i<nhidden;i++)
    h_pre[i] = h[i];
}

// compute the output layer's  values
for(i = 0; i < OUT_NUM; i++)
    {
    sum = 0;
    for(j = 0;j < nhidden; j++)
        sum += w_output[i][j] * h[j];
    out[i] = sigmoid(sum + b_output[i]);
    }

free(h);
return out;
}