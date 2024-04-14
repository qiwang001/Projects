#include"stdlib.h"
#include"stdio.h"
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include "omp.h"
#include "mpi.h"
#include "assert.h"
//#include <cuda.h>
//#include <curand.h>
//#include<curand_kernel.h>
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
double tt1 =omp_get_wtime();
int nprocs; 
int mype; 
int stat;   
MPI_Init(&argc, &argv);
stat = MPI_Comm_size(MPI_COMM_WORLD, &nprocs); 
assert (stat == MPI_SUCCESS);
stat = MPI_Comm_rank(MPI_COMM_WORLD, &mype); 
assert (stat == MPI_SUCCESS);

int m=1000;
long n = atoi(argv[1])/nprocs;
int ngrid = atoi(argv[2]);
int nblocks = atoi(argv[3]);
int nthreads = atoi(argv[4]);


double** grid_h = grid_maker(m);
double** grid_temp = grid_maker(m);
unsigned long long nsample=0;
unsigned long long * sample_h=&nsample;
unsigned long long nsample_total=0;
float time_host=1;
mpi_call(n,nblocks,nthreads,ngrid,grid_h,sample_h,&time_host,m, mype);
printf("node %d kernel time: %fs \n", mype, time_host/1000);
printf("node %d sample: %llu\n",mype, nsample);


if(mype==0)
{
nsample_total += nsample;
for (int i=1;i<nprocs;++i)
{
MPI_Recv(&nsample, 1, MPI_UNSIGNED_LONG_LONG, i, 99, MPI_COMM_WORLD,&stat);
nsample_total+=nsample;
MPI_Recv(grid_temp[0], m * m, MPI_DOUBLE, i, 100, MPI_COMM_WORLD, &stat);
for(int i=0; i<m; i++)
    for(int j=0; j<m; j++)
    grid_h[i][j] += grid_temp[i][j];

}
}

else
{
MPI_Send(&nsample, 1, MPI_UNSIGNED_LONG_LONG, 0, 99, MPI_COMM_WORLD);
MPI_Send(grid_h[0], m * m, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
}

if(mype == 0){
FILE* fp = fopen("grid.txt","w");
if(fp == NULL) 
{
printf("Error opening file.\n");
return 1;
}
for(int i=0;i<m;i++)
    {
    for(int j=0;j<m;j++)
        fprintf(fp, "%f ",grid_h[i][j]);
    fprintf(fp, "\n");
    }
}
double tt2 =omp_get_wtime();

if (mype==0)
{
printf("total time %fs,\n ", tt2-tt1);
printf("total sample made %llu,\n ", nsample_total);
}

free(grid_h[0]);
free(grid_h);
free(grid_temp[0]);
free(grid_temp);
MPI_Finalize();
return 0;

}
