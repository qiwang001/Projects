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
int m1 = atoi(argv[2]);
int nt=atoi(argv[3]);
/*int N = 1000000;
int m1=1000;
int nt=1;v*/
double** grid = grid_maker(m1);
//double grid[1000][1000][8];
srand(time(NULL));
unsigned int seed = time(NULL);
unsigned seed = time(NULL);
double t1 = omp_get_wtime();
long sample=0;
int i=0;
#pragma omp parallel default(none) num_threads(nt) shared(seed,grid,N) //reduction(+ : sample)
{
    // Padding data to avoid false sharing
    double wy=10, wmax=10, r=6, len_s=0,len_n=0, wx, wz,b;
    double c[3] = {0,12,0};
    double I[3];
    double L[3] = {4,4,-1};
    double s[3];
    double n[3];
    double phi,cosine_theta;
    int p,q,m=1000;
#pragma omp for 
    for(int i=0;i<=N;i++)
    {   //if(i%1000==0)
        //printf("got here i=%d",i);
        while(true)
        {
        //#pragma omp atomic
        //sample += 1;
        phi = PI* (double)rand_r(&seed)/RAND_MAX;
        cosine_theta = ((double)rand_r(&seed)/RAND_MAX - 0.5)*2;
        //sine_theta = sqrt(1-cosine_theta*cosine_theta);
        /*
        view_light[0] = sqrt(1-cosine_theta*cosine_theta) * cos(phi);
        view_light[1] = sqrt(1-cosine_theta*cosine_theta) * sin(phi);
        view_light[2] = cosine_theta;*/

        wx = sqrt(1-cosine_theta*cosine_theta)* cos(phi) *  wy/(sqrt(1-cosine_theta*cosine_theta) * sin(phi));
        wz = cosine_theta *  wy/(sqrt(1-cosine_theta*cosine_theta) * sin(phi));
        //xx = pow(sqrt(1-cosine_theta*cosine_theta) * sin(phi)*c[1],2) +r*r -(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]);

        if(abs(wx) <= wmax && abs(wz) <=wmax && pow(sqrt(1-cosine_theta*cosine_theta) * sin(phi)*c[1],2) + r*r -(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]) > 0)
            break;
        }
        //t = sqrt(1-cosine_theta*cosine_theta) * cos(phi)*c[0]+sqrt(1-cosine_theta*cosine_theta) * sin(phi)*c[1]+cosine_theta*c[2] - sqrt(sqrt(1-cosine_theta*cosine_theta) * cos(phi)*c[0]+sqrt(1-cosine_theta*cosine_theta) * sin(phi)*c[1]+cosine_theta*c[2]*sqrt(1-cosine_theta*cosine_theta) * cos(phi)*c[0]+sqrt(1-cosine_theta*cosine_theta) * sin(phi)*c[1]+cosine_theta*c[2] +r*r -(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]));    
        I[0] = sqrt(1-cosine_theta*cosine_theta) * cos(phi)*c[0]+sqrt(1-cosine_theta*cosine_theta) * sin(phi)*c[1]+cosine_theta*c[2] - sqrt(sqrt(1-cosine_theta*cosine_theta) * cos(phi)*c[0]+sqrt(1-cosine_theta*cosine_theta) * sin(phi)*c[1]+cosine_theta*c[2]*sqrt(1-cosine_theta*cosine_theta) * cos(phi)*c[0]+sqrt(1-cosine_theta*cosine_theta) * sin(phi)*c[1]+cosine_theta*c[2] +r*r -(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]))*sqrt(1-cosine_theta*cosine_theta) * cos(phi);
        s[0] = L[0] - I[0];
        n[0] = I[0] - c[0];
        I[1] = sqrt(1-cosine_theta*cosine_theta) * cos(phi)*c[0]+sqrt(1-cosine_theta*cosine_theta) * sin(phi)*c[1]+cosine_theta*c[2] - sqrt(sqrt(1-cosine_theta*cosine_theta) * cos(phi)*c[0]+sqrt(1-cosine_theta*cosine_theta) * sin(phi)*c[1]+cosine_theta*c[2]*sqrt(1-cosine_theta*cosine_theta) * cos(phi)*c[0]+sqrt(1-cosine_theta*cosine_theta) * sin(phi)*c[1]+cosine_theta*c[2] +r*r -(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]))*sqrt(1-cosine_theta*cosine_theta) * sin(phi);
        s[1] = L[1] - I[1];
        n[1] = I[1] - c[1];
        I[2] = sqrt(1-cosine_theta*cosine_theta) * cos(phi)*c[0]+sqrt(1-cosine_theta*cosine_theta) * sin(phi)*c[1]+cosine_theta*c[2] - sqrt(sqrt(1-cosine_theta*cosine_theta) * cos(phi)*c[0]+sqrt(1-cosine_theta*cosine_theta) * sin(phi)*c[1]+cosine_theta*c[2]*sqrt(1-cosine_theta*cosine_theta) * cos(phi)*c[0]+sqrt(1-cosine_theta*cosine_theta) * sin(phi)*c[1]+cosine_theta*c[2] +r*r -(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]))*cosine_theta;
        s[2] = L[2] - I[2];
        n[2] = I[2] - c[2];

        len_s = sqrt(s[0]*s[0]+s[1]*s[1] + s[2]*s[2]);
        len_n = sqrt(n[0]*n[0] +n[1]*n[1]+n[2]*n[2]);
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
