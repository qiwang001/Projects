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
float** grid_maker(int n)
{
    float* g = (float *)calloc(n*n, sizeof(float));
    float ** M = (float**)malloc(sizeof(float*)*n);
    for(int i=0;i<n;i++)
        M[i] = g + i * n;
    return M;
}

int
main(int argc, char**argv)
//main()
{
int N = atoi(argv[1]);
int m1 = atoi(argv[2]);
int nt=atoi(argv[3]);
/*int N = 1000000;
int m1=1000;
int nt=1;*/
float** grid = grid_maker(m1);
//double grid[1000][1000][8];
srand(time(NULL));
unsigned seed= time(NULL);
double t1 = omp_get_wtime();
long sample=0;
int i=0;
#pragma omp parallel default(none) num_threads(nt) shared(N,grid)
{
    float wy=10, d1[32], wmax=10,d2[32], r=6,d3[32], len_s=0,d4[32],
        len_n=0,d5[32],wx,d6[32], wz,d7[32],temp,d8[32],t,d0[32],b,d10[32],xx,d11[32];
        float c[3] = {0,12,0};
        float d12[32];
        float I[3];
        float d13[32];
        float L[3] = {4,4,-1};
        float d14[32];
        float s[3];
        float d15[32];
        float n[3];
        float d16[32];
        float view_light[3],d17[32], phi,d18[32],d19[32],cosine_theta,d20[32], sine_theta;
        int p,q,m=1000;
#pragma omp for 
    for(int i=0;i<=N;i++)
    { 
        while(true)
        {
        //#pragma omp atomic
        //sample += 1;
        phi = PI* (float)rand()/RAND_MAX;
        cosine_theta = ((float)rand()/RAND_MAX - 0.5)*2;
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
for(int i=0;i<m1;i++)
    {
    for(int j=0;j<m1;j++)
        fprintf(fp, "%f ",grid[i][j]);
    fprintf(fp, "\n");
    }

free(grid[0]);
free(grid);
printf("\nsample: %d",sample);
return 0;
}
