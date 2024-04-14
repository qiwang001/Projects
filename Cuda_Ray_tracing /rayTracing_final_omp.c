#include"stdlib.h"
#include"stdio.h"
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include "omp.h"
#define MAX(a,b) ((a)>(b)? (a):(b))
#define MIN(a,b) ((a)<(b)? (a):(b))
#define ABS(a) ((a)>0? (a):(-a))
#define PI 3.1415926
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
//main()
{
int N = atoi(argv[1]);
int m = atoi(argv[2]);
int nt=atoi(argv[3]);
/*int N = 1000000;
int m1=1000;
int nt=1;*/
double** grid = grid_maker(m);
//double grid[1000][1000][8];
srand(time(NULL));
unsigned seed= time(NULL);
double t1 = omp_get_wtime();
long sample=0;
int i=0;
#pragma omp parallel default(none) num_threads(nt) shared(seed, N,grid) firstprivate(m) reduction(+ : sample)
{
// padd data to avoid false sharing
double wy=10, d1[32], wmax=10,d2[32], r=6,d3[32], len_s=0,d4[32],
len_n=0,d5[32],wx,d6[32], wz,d7[32],temp,d8[32],t,d0[32],b,d10[32],xx,d11[32];
double c[3] = {0,12,0};
double d12[32];
double I[3];
double d13[32];
double L[3] = {4,4,-1};
double d14[32];
double s[3];
double d15[32];
double n[3];
double d16[32];
double view_light[3],d17[32], phi,d18[32],d19[32],cosine_theta,d20[32], sine_theta;
int p,q;
#pragma omp for 
    for(int i=0;i<=N;i++)
    { 
        while(true)
        {
        #pragma omp atomic
        sample += 1;
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

        if(abs(wx) <= wmax && abs(wz) <=wmax && xx > 0)
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

        b = MAX(0, b);
        q = m*(wx + wmax)/(2*wmax);
        p = m*(wz + wmax)/(2*wmax);
        //#pragma omp atomic
        grid[q][p] = grid[q][p] + b;
    }

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
printf("\nsample: %d",sample);
return 0;
}
