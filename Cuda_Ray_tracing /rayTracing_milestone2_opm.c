#include"stdlib.h"
#include"stdio.h"
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include "omp.h"
# define PI 3.1415926
double** grid_maker(int n)
{
    double* g = (double *)calloc(n*n, sizeof(double));
    double ** M = (double**)malloc(sizeof(double*)*n);
    for(int i=0;i<n;i++)
        M[i] = g + i * n;
    return M;
}

int
main(int argc, char**argv)

{
int m = 1000;
int N = atoi(argv[1]);
int nt=atoi(argv[2]);
double** grid = grid_maker(m);
srand(time(NULL));
unsigned int seed = 132989120;
double t1 = omp_get_wtime();
#pragma omp parallel for default(none) shared(N,seed,grid) schedule(guided) num_threads(nt)
for (int i=0;i<=N;i++)
    {   
        double wy=10, wmax=10, r=6, len_s=0, 
        len_n=0,wx, wz,temp,t,b,xx,mm;
        double c[3] = {0,12,0};
        double I[3];
        double L[3] = {4,4,-1};
        double s[3];
        double n[3];
        double view_light[3],phi,cosine_theta, sine_theta;
        int p,q,m=1000;
        while(true)
        {
        phi = PI* (double)rand_r(&seed)/RAND_MAX;
        cosine_theta = ((double)rand_r(&seed)/RAND_MAX - 0.5)*2;
        sine_theta = sqrt(1-cosine_theta*cosine_theta);
        view_light[0] = sine_theta * cos(phi);
        view_light[1] = sine_theta * sin(phi);
        view_light[2] = cosine_theta;

        wx = view_light[0] *  wy/view_light[1];
        wz = view_light[2] *  wy/view_light[1];

        temp=view_light[0]*c[0]+view_light[1]*c[1]+view_light[2]*c[2];
        xx = temp*temp +r*r -(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]);

        if(fabs(wx) <= wmax && fabs(wz) <=wmax && xx > 0)
            break;
        }
        t = temp - sqrt(xx);
        len_s=0;
        len_n=0;
        for(int i=0;i<3;i++)
            {
            I[i] = t*view_light[i];
            s[i] = L[i] - I[i];
            n[i] = I[i] - c[i];
            len_s+=s[i]*s[i];
            len_n+=n[i]*n[i];
            }
        len_s = sqrt(len_s);
        len_n = sqrt(len_n);
        b=0;
        for(int i=0;i<3;i++)
            {
            s[i] = s[i]/len_s;
            n[i] = n[i]/len_n;
            b+=s[i]*n[i];
            }

        b = fmax(0, b);
        q = m*(wx + wmax)/(2*wmax);
        p = m*(wz + wmax)/(2*wmax);
        //#pragma omp atomic
        grid[q][p] = grid[q][p] + b;
    }

double t2 = omp_get_wtime();
printf("time used: %f", t2-t1);

FILE* fp = fopen("grid.txt","w");
if(fp == NULL) 
{
printf("Error opening file.\n");
return 1;
}
for(int i=0;i<m;i++)
    {
    for(int j=0;j<m;j++)
        fprintf(fp, "%f ",grid[i][j]);
    fprintf(fp, "\n");
    }

free(grid[0]);
free(grid);
return 0;
}
