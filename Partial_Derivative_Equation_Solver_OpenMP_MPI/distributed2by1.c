#include "stdio.h"
#include "stdlib.h"
#include "assert.h"
#include "mpi.h"
#include"omp.h"
#include"math.h"
//export OMPI_CC=gcc-13
double**
data_half_matrix(int n)
{
    double *data = (double*)malloc(n * (n/2) * sizeof(double));
    double **M = (double**)malloc((n/2) * sizeof(double*)); 
    for (int i=0; i < n/2; i++)
        M[i] = data + n * i;
        //M[i] = &data[n * i];
    return M;
}

int
main(int argc, char** argv)
{
int stat, nprocs, mype,im1,ia1,jm1,ja1;
double t1,t2,t3,t4,grind_rate,time_used,x,y,u,v;
int  M=8000;
double **pt0,**pt1;
double **c = data_half_matrix(4000);
double **c_new = data_half_matrix(4000);
double **c1 = data_half_matrix(4000);
double **c1_new = data_half_matrix(4000);
double* guard0 = (double*)malloc(sizeof(double)*M);
double* tosend0 = (double*)malloc(sizeof(double)*M);
double* guard1 = (double*)malloc(sizeof(double)*M);
double* tosend1 = (double*)malloc(sizeof(double)*M);
double* parameter_double = (double*)malloc(sizeof(double)*4);
int* parameter_int = (int*)malloc(sizeof(int)*3);
double* parameter_double1 = (double*)malloc(sizeof(double)*4);
int* parameter_int1 = (int*)malloc(sizeof(int)*3);


MPI_Status status;
MPI_Init(&argc,&argv);
stat=MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
assert(stat==MPI_SUCCESS);
stat=MPI_Comm_rank(MPI_COMM_WORLD, &mype);
assert(stat==MPI_SUCCESS);

if(mype==0)
{
FILE *data=fopen("lax0.txt", "w");
if(data==NULL)
    {
    printf("Error opening file!\n");
    return 1;  // Indicate failure
    }
double L=1, T=1, dt=1.25e-4, dx;
int nt=1, N=4000,NT = 1/dt;
dx=L/(N-1);
parameter_double[0] =L;
parameter_double[1]=T;
parameter_double[2]=dt;
parameter_double[3]=dx;
parameter_int[0]=nt;
parameter_int[1]=N;
parameter_int[2]=NT;

MPI_Send(parameter_double,4,MPI_DOUBLE,1,10,MPI_COMM_WORLD);
MPI_Send(parameter_int,3,MPI_DOUBLE,1,20,MPI_COMM_WORLD);


//initiate the 1s in c[][]
for(int i = 2*N/5; i<N/2; i++)
    for(int j=0; j<N; j++)
        c[i][j] = 1;

//write initial data to file
for(int i = 0; i<N/2; i++)
    { 
    for(int j=0; j<N; j++)  
        fprintf(data,"%.4f ",c[i][j]);
    fprintf(data,"\n");
    }


t1 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    
    if(n%100==0)
    {
        printf("going well on %d \n",n);
        printf("time passed: %fs\n", omp_get_wtime()-t1);
    }
    
    //prepare data to send to process #1
    for(int j=0;j<N;j++)
        {
        tosend0[j]=c[0][j];
        tosend0[N+j]=c[N/2-1][j];
        }
    //MPI COMM
    MPI_Recv(guard0, 2*N,MPI_DOUBLE,1,n,MPI_COMM_WORLD,&status);
    MPI_Send(tosend0,2*N,MPI_DOUBLE,1,n,MPI_COMM_WORLD);
    //compute c_new[0][*]. Need to use c[N-1][*], which is stored  
    //in guard0[N:2N]
    x= -L/2;
    v=(-1.414)*x;
    for(int j=1;j<N-1;j++)
    {
    y = -L/2.0 + j*dx;
    u = 1.414 * y;
    c_new[0][j]=.25*(guard0[N+j] + c[1][j] + c[0][j-1] + c[0][j+1]) 
    - dt/(2*dx) * (u * (c[1][j]-guard0[N+j]) + v * (c[0][j+1]-c[0][j-1]));
    }

    y = -L/2.0;
    u = 1.414 * y;
    c_new[0][0]=.25*(guard0[N] + c[1][0] + c[0][N-1] + c[0][1])
    - dt/(2*dx) * (u* (c[1][0]-guard0[N]) + v* (c[0][1]-c[0][N-1]));

    y = -L/2.0 + dx*(N-1);
    u = 1.414 * y;
    c_new[0][N-1]=.25*(guard0[2*N-1] + c[1][N-1] + c[0][N-2] + c[0][0]) 
    - dt/(2*dx) * (u * (c[1][N-1]-guard0[2*N-1]) + v * (c[0][0]-c[0][N-2]));

    //compute c_new from row 1 to row N/2-2
    for(int i=1;i<N/2-1;i++)
       { 
        x= -L/2.0 + i*dx;
        v= -1.414 * x;
        for(int j=0;j<N;j++)
            {
            jm1=(j-1)%N;
            if (jm1<0)
                jm1+=N;
            ja1 = (j + 1)%N;
            y = -L/2.0 + j*dx;
            u = 1.414 * y;
            c_new[i][j]=.25*(c[i-1][j] + c[i+1][j] + c[i][jm1] + c[i][ja1]) + 
            - dt/(2*dx) * (u * (c[i+1][j]-c[i-1][j]) + v * (c[i][ja1]-c[i][jm1]));
            }
        }


    //compute c_new[N/2-1][*]. Need to use c[N/2][*] which is stored
    //in guard0[0:N]
    x = -L/2 + (N/2 - 1)*dx;
    v = -1.414 * x;
    for(int j=1;j<N-1;j++)
        {   
            y = -L/2.0 + j*dx;
            u = 1.414 * y;
            c_new[N/2 - 1][j]=.25*(c[N/2 - 2][j] + guard0[j] + c[N/2 - 1][j-1] + c[N/2 - 1][j+1]) 
            - dt/(2*dx) * (u * (guard0[j]-c[N/2 - 2][j]) + v * (c[N/2 - 1][j+1]-c[N/2 - 1][j-1]));
        }
    y = -L/2.0 ;
    u = 1.414 * y;

    c_new[N/2 - 1][0] = .25*(c[N/2 - 2][0] + guard0[0] + c[N/2 - 1][N-1] + c[N/2 - 1][1]) 
    - dt/(2*dx) * (u * (guard0[0] - c[N/2 - 2][0]) + v * (c[N/2 - 1][1]-c[N/2 - 1][N-1]));

    y = -L/2.0 + (N-1)*dx;
    u = 1.414 * y;
    c_new[N/2 - 1][N-1] = .25*(c[N/2 - 2][N-1] + guard0[N-1] + c[N/2 - 1][N-2] + c[N/2 - 1][0]) 
    - dt/(2*dx) * (u * (guard0[N-1]-c[N/2 - 2][N-1]) + v * (c[N/2 - 1][0]-c[N/2 - 1][N-2]));
    /*
    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/2;i++)
            {
            for(int j=0;j<N;j++)
                fprintf(data,"%.4f ",c_new[i][j]); 
            fprintf(data,"\n"); 
            }
       } */

    pt0=c;
    c=c_new;
    c_new=pt0;
    }
    t2 = omp_get_wtime();
    fclose(data);
}

else if(mype==1)
{
FILE *data1=fopen("lax1.txt", "w");
if(data1==NULL)
    {
    printf("Error opening file!\n");
    return 1;  // Indicate failure
    }

MPI_Recv(parameter_double1, 4,MPI_DOUBLE,0,10,MPI_COMM_WORLD,&status);
MPI_Recv(parameter_int1, 3,MPI_DOUBLE,0,20,MPI_COMM_WORLD,&status);

double L, T, dt, dx;
int nt, N,NT;
dx=L/(N-1);
L=parameter_double1[0];
T=parameter_double1[1];
dt=parameter_double1[2];
dx=parameter_double1[3];
nt=parameter_int1[0];
N=parameter_int1[1];
NT=parameter_int1[2];
//initiate c1 values
for(int i = 0; i < N/10; i++)
    for(int j=0; j<N; j++)
        c1[i][j] = 1;
     
//write initial data to file
for(int i = 0; i<N/2; i++)
    { 
    for(int j=0; j<N; j++)  
        fprintf(data1,"%.4f ",c1[i][j]);
    fprintf(data1,"\n");
    }
//}
//#pragma omp parallel for default(none)  shared(data,c,c1,N,NT,dx,dt,u,v,im1,ia1,jm1,ja1,x,y) schedule(static) num_threads(nt)
    
t3 = omp_get_wtime();
//#pragma omp parallel for default(none) shared(data,c,c1,N,NT,dx,dt,u,v,t,L,x) private(n,i,j,im1,ia1,jm1,ja1,y) schedule(static) num_threads(nt)
for(int n=2;n <= NT;n++)
    {  
        /*if(n%100==0)
            {
            std::cout<<"NT is "<<NT<<" going well on "<<n<<"th iteration.."<<endl;
            std::cout<<"time eclipsed: "<<omp_get_wtime()-t1<<endl;
            }*/
    //prepare date to send to process 0         
    for(int j=0;j<N;j++)
        {
        tosend1[j]=c1[0][j];
        tosend1[N+j]=c1[N/2-1][j];
        }
    
    MPI_Send(tosend1, 2*N,  MPI_DOUBLE,0,n,MPI_COMM_WORLD);
    MPI_Recv(guard1,  2*N, MPI_DOUBLE,0,n,MPI_COMM_WORLD,&status);

    //compute c1_new[0][*]. Need to use c0[N/2-1][*], which is stored in guard1[N:2N]
    x=-L/2+dx*(N/2);
    v=(-1.414)*x;
    for(int j=1;j<N-1;j++)
    {
    y = -L/2.0 + j*dx;
    u = 1.414 * y;
    c1_new[0][j]=.25*(guard1[N+j] + c1[1][j] + c1[0][j-1] + c1[0][j+1]) 
    - dt/(2*dx) * (u * (c1[1][j]-guard1[N+j]) + v * (c1[0][j+1]-c1[0][j-1]));
    }


    y = -L/2.0;
    u = 1.414 * y;
    c1_new[0][0]=.25*(guard1[N] + c1[1][0] + c1[0][N-1] + c1[0][1])
    - dt/(2*dx) * (u * (c1[1][0]-guard1[N]) + v * (c1[0][1]-c1[0][N-1]));


    y = -L/2.0 + dx*(N-1);
    u = 1.414 * y;
    c1_new[0][N-1]=.25*(guard1[2*N-1] + c1[1][N-1] + c1[0][N-2] + c1[0][0]) 
    - dt/(2*dx) * (u * (c1[1][N-1]-guard1[2*N-1]) + v * (c1[0][0]-c1[0][N-2]));

    //Compute c1_new[1][*] to c1_new[N/2-2][*]
    for(int i=1;i<N/2-1; i++)
    {
        x= -L/2.0 + (i + N/2)*dx;
        v= -1.414 * x;
        for(int j=0;j < N;j++)
            {
            jm1=(j-1)%N;
            if (jm1<0)
                jm1+=N;
            ja1 = (j+1)%N;
            y = -L/2.0 + j*dx;
            u = 1.414 * y;
            c1_new[i][j]=.25*(c1[i-1][j] + c1[i+1][j] + c1[i][jm1] + c1[i][ja1]) + 
            - dt/(2*dx) * (u * (c1[i+1][j]-c1[i-1][j]) + v * (c1[i][ja1]-c1[i][jm1]));
            //if(n==NT/2||n==NT)
            //    fprintf(data,"%.4f ",c[i][j]);
            }
        //if(n==NT/2||n==NT)
        //    fprintf(data,"\n");
    }

    //compute c1_new[N/2-1][*]. Need to use c[0][*], which is stored in guard1[0:N]
    x = -L/2 + (N - 1)*dx;
    v = -1.414 * x;
    for(int j=1;j<N-1;j++)
        {   
            y = -L/2.0 + j*dx;
            u = 1.414 * y;
            c1_new[N/2 - 1][j]=.25*(c1[N/2 - 2][j] + guard1[j] + c1[N/2 - 1][j-1] + c1[N/2 - 1][j+1]) 
            - dt/(2*dx) * (u * (guard1[j]-c1[N/2 - 2][j]) + v * (c1[N/2 - 1][j+1]-c1[N/2 - 1][j-1]));
        }
    
    y = -L/2.0;
    u = 1.414 * y;
    c1_new[N/2-1][0]=.25*(c1[N/2-2][0] + guard1[0] + c1[N/2 - 1][N-1] + c1[N/2-1][1])
    - dt/(2*dx) * (u * (guard1[0]-c1[N/2-2][0]) + v * (c1[N/2-1][1]-c1[N/2-1][N-1]));

    y = -L/2.0 + dx*(N-1);
    u = 1.414 * y;
    c1_new[N/2-1][N-1]=.25*(c1[N/2-2][N-1] + guard1[N-1] + c1[N/2 - 1][N-2] + c1[N/2-1][0])
    - dt/(2*dx) * (u * (guard1[N-1]-c1[N/2-2][N-1]) + v * (c1[N/2-1][0]-c1[N/2-1][N-2]));
    /*
    if(n==NT/2||n==NT)
        {
        for(int i=0;i<N/2;i++)
            {
            for(int j=0;j<N;j++)
                fprintf(data1,"%.4f ",c1_new[i][j]);
            fprintf(data1,"\n");        
            }
        }*/

    pt1=c1;
    c1=c1_new;
    c1_new=pt1;
    }
t4 = omp_get_wtime();
fclose(data1);
}
MPI_Finalize();

double Nd=4000, NTd=1/(1.25e-4);
if(mype==0)
printf("Time used: %f, grind_rate: %.2fM\n", t2-t1, (Nd*Nd*NTd/(t2-t1))/1000000);
if(mype==1)
printf("Time used: %f, grind_rate: %.2fM\n", t4-t3, (Nd*Nd*NTd/(t4-t3))/1000000);
free(c[0]);
free(c1[0]);
free(c_new[0]);
free(c1_new[0]);
free(c);
free(c1);
free(c_new);
free(c1_new);
free(guard0);
free(guard1);
free(tosend0);
free(tosend1);
return 0;
}
