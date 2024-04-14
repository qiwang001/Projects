#include "stdio.h"
#include "stdlib.h"
#include "assert.h"
#include "mpi.h"
#include"omp.h"
#include"math.h"
//export OMPI_CC=gcc-13
double**
data_quarter_matrix(int n)
{
    double *data = (double*)malloc((n/2)* (n/2) * sizeof(double));
    double **M = (double**)malloc((n/2) * sizeof(double*)); 
    for (int i=0; i < n/2; i++)
        M[i] = data + n/2 * i;
        //M[i] = &data[n * i];
    return M;
}

int
main(int argc, char** argv)
{
int stat, nprocs, mype,im1,ia1,jm1,ja1;
double t1,t2,t3,t4,t5,t6,t7,t8,grind_rate,time_used,
x,y,u,v;
int nt=atoi(argv[2]),N=atoi(argv[1]),NT=8000;
double **pt0,**pt1,**pt2, **pt3;
double **c0 = data_quarter_matrix(N);
double **c0_new = data_quarter_matrix(N);
double **c1 = data_quarter_matrix(N);
double **c1_new = data_quarter_matrix(N);
double **c2 = data_quarter_matrix(N);
double **c2_new = data_quarter_matrix(N);
double **c3 = data_quarter_matrix(N);
double **c3_new = data_quarter_matrix(N);
double* guard0from1 = (double*)malloc(sizeof(double)*N);
double* guard0from2 = (double*)malloc(sizeof(double)*N);
double* send0to1 = (double*)malloc(sizeof(double)*N);
double* send0to2 = (double*)malloc(sizeof(double)*N);
double* guard1from0 = (double*)malloc(sizeof(double)*N);
double* guard1from3 = (double*)malloc(sizeof(double)*N);
double* send1to0 = (double*)malloc(sizeof(double)*N);
double* send1to3 = (double*)malloc(sizeof(double)*N);
double* guard2from0 = (double*)malloc(sizeof(double)*N);
double* guard2from3 = (double*)malloc(sizeof(double)*N);
double* send2to0 = (double*)malloc(sizeof(double)*N);
double* send2to3 = (double*)malloc(sizeof(double)*N);
double* guard3from1 = (double*)malloc(sizeof(double)*N);
double* guard3from2 = (double*)malloc(sizeof(double)*N);
double* send3to1 = (double*)malloc(sizeof(double)*N);
double* send3to2 = (double*)malloc(sizeof(double)*N);
double* parameter_double = (double*)malloc(sizeof(double)*4);
double* parameter_double1 = (double*)malloc(sizeof(double)*4);
double* parameter_double2 = (double*)malloc(sizeof(double)*4);
double* parameter_double3 = (double*)malloc(sizeof(double)*4);


MPI_Status status;
MPI_Init(&argc,&argv);
stat=MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
assert(stat==MPI_SUCCESS);
stat=MPI_Comm_rank(MPI_COMM_WORLD, &mype);
assert(stat==MPI_SUCCESS);

if(mype==0)
{
FILE *data0=fopen("lax0.txt", "w");
if(data0==NULL)
    {
    printf("Error opening file!\n");
    return 1;  // Indicate failure
    }

double L=1, T=1, dt=1.25e-4, dx;
dx=L/(N-1);
parameter_double[0] =L;
parameter_double[1]=T;
parameter_double[2]=dt;
parameter_double[3]=dx;
MPI_Send(parameter_double,4,MPI_DOUBLE,1,10,MPI_COMM_WORLD);
MPI_Send(parameter_double,4,MPI_DOUBLE,2,10,MPI_COMM_WORLD);
MPI_Send(parameter_double,4,MPI_DOUBLE,3,10,MPI_COMM_WORLD);

//initiate the 1s in c0[][]
for(int i = 2*N/5; i<N/2; i++)
    for(int j=0; j<N/2; j++)
        c0[i][j] = 1;

//write initial data to file
for(int i = 0; i<N/2; i++)
    { 
    for(int j=0; j<N/2; j++)  
        fprintf(data0,"%.4f ",c0[i][j]);
    fprintf(data0,"\n");
    }


t1 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #1 and #2
    if(n%100==0)
    {
        printf("going well on %dth\n ",n);
        printf("time used: %f\n", omp_get_wtime()-t1);
    }
    for(int j=0;j<N/2;j++)
        {
        send0to1[j]=c0[j][0];
        send0to1[N/2+j]=c0[j][N/2-1];

        send0to2[j]=c0[0][j];
        send0to2[N/2+j]=c0[N/2-1][j];
        }
    //MPI COMM with #1
    MPI_Recv(guard0from1, N,MPI_DOUBLE,1,n,MPI_COMM_WORLD,&status);
    MPI_Send(send0to1,N,MPI_DOUBLE,1,n,MPI_COMM_WORLD);

    //MPI COMM with #2
    MPI_Recv(guard0from2, N,MPI_DOUBLE,2,n,MPI_COMM_WORLD,&status);
    MPI_Send(send0to2,N,MPI_DOUBLE,2,n,MPI_COMM_WORLD);

    //compute c0_new[0][j].for j=1,2,..,N/2-2, c2[N/2-1][*]
    //c0[-1][j] = guard0from2[N/2 + j]
    x= -L/2;
    v=(-1.414)*x;
    for(int j=1;j<N/2-1;j++)
    {
    y = -L/2.0 + j*dx;
    u = 1.414 * y;
    c0_new[0][j]=.25*(guard0from2[N/2+j] + c0[1][j] + c0[0][j-1] + c0[0][j+1]) 
    - dt/(2*dx) * (u * (c0[1][j]-guard0from2[N/2+j]) + v * (c0[0][j+1]-c0[0][j-1]));
    }

    //compute c0_new[0][0] , need to use c[-1][0] = guard0from2[N/2] and c[0][-1]=guard0from1[N/2]
    y = -L/2.0;
    u = 1.414 * y;
    c0_new[0][0]=.25*(guard0from2[N/2] + c0[1][0] + guard0from1[N/2] + c0[0][1])
    - dt/(2*dx) * (u* (c0[1][0]-guard0from2[N/2]) + v* (c0[0][1]-guard0from1[N/2]));

    //compute c0_new[0][N/2-1], need to use c[0][N/2]=guard0from1[0], c[-1][N/2-1]=guard0from2[N-1]
    y = -L/2.0 + dx*(N/2-1);
    u = 1.414 * y;
    c0_new[0][N/2-1]=.25*(guard0from2[N-1] + c0[0][1] + c0[0][N/2-2] + guard0from1[0])
    - dt/(2*dx) * ((u*(c0[1][N/2-1] - guard0from2[N-1])) + v* (guard0from1[0] - c0[0][N/2-2]));
    
    //compute c0_new from row 1 to row N/2-2
    #pragma omp parallel for default(none) shared(L,N,c0,c0_new,guard0from1,dx,dt) private(x,y,u,v) schedule(static) num_threads(nt)
    for(int i=1;i<N/2-1;i++)
       { 
        x= -L/2.0 + i*dx;
        v= -1.414 * x;

        //compute c_new[i][0]
        y= -L/2; 
        u= 1.414*y;
        c0_new[i][0]=.25*(c0[i-1][0] + c0[i+1][0] + guard0from1[N/2+i] + c0[i][1])
        - dt/(2*dx) * (u * (c0[i+1][0]-c0[i-1][0]) + v * (c0[i][1]-guard0from1[N/2+i]));
        //compute c0_new[i][j] for j=1,2..,N/2-2.
        for(int j=1;j<N/2-1;j++)
            {
            y = -L/2.0 + j*dx;
            u = 1.414 * y;
            c0_new[i][j]=.25*(c0[i-1][j] + c0[i+1][j] + c0[i][j-1] + c0[i][j+1]) 
            - dt/(2*dx) * (u * (c0[i+1][j]-c0[i-1][j]) + v * (c0[i][j+1]-c0[i][j-1]));
            }

        //compute c0_new[i][N/2-1]
        y=-L/2 + (N/2-1)*dx;
        u = 1.414*y;
        c0_new[i][N/2-1]=.25*(c0[i-1][N/2-1] + c0[i+1][N/2-1] + c0[i][N/2-2] + guard0from1[i]) 
        - dt/(2*dx) * (u * (c0[i+1][N/2-1]-c0[i-1][N/2-1]) + v * (guard0from1[i]-c0[i][N/2-2]));

        }

        //compute c0_new[N/2-1][*]. Need to use c[N/2][j] = guard0from2[j] for j = 1,2..N/2-2
        x = -L/2 + (N/2 - 1)*dx;
        v = -1.414 * x;
        for(int j=1; j<N/2-1; j++)
            {
            y = -L/2.0 + j*dx;
            u = 1.414 * y;
            c0_new[N/2-1][j]=.25*(c0[N/2 - 2][j] + guard0from2[j] + c0[N/2 - 1][j-1] + c0[N/2 - 1][j+1])
            -dt/(2*dx) * (u * (guard0from2[j]-c0[N/2 - 2][j]) + v * (c0[N/2 - 1][j+1]-c0[N/2 - 1][j-1]));
            }
        
        //compute c0_new[N/2-1][0], c0[N/2][0]=guard0from2[0], c0[N/2-1][-1]=guard0from1[N-1]
        y = -L/2.0;
        u = 1.414 * y;
        c0_new[N/2-1][0]=.25*(c0[N/2 - 2][0] + guard0from2[0] + guard0from1[N-1] + c0[N/2 - 1][1])
        -dt/(2*dx) * (u * (guard0from2[0]-c0[N/2 - 2][0]) + v * (c0[N/2 - 1][1]-guard0from1[N-1]));
        

        //c0[N/2-1][N/2] = guard0from1[N/2-1], c0[N/2][N/2-1]=guard0from2[N/2-1]
        y = -L/2.0 + (N/2-1)*dx;
        u = 1.414 * y;
        c0_new[N/2-1][N/2-1]=.25*(c0[N/2 - 2][N/2-1] + guard0from2[N/2-1] + c0[N/2-1][N/2-2] + guard0from1[N/2-1])
        -dt/(2*dx) * (u * (guard0from2[N/2-1]-c0[N/2 - 2][N/2-1]) + v * (guard0from1[N/2-1]-c0[N/2-1][N/2-2]));
        
    
    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/2;i++)
            {
            for(int j=0;j<N/2;j++)
                fprintf(data0,"%.4f ",c0_new[i][j]); 
            fprintf(data0,"\n"); 
            }
       } 

    pt0=c0;
    c0=c0_new;
    c0_new=pt0;
}
t2 = omp_get_wtime();
fclose(data0);
}

else if(mype==1)
{
FILE *data1=fopen("lax1.txt", "w");
if(data1==NULL)
    {
    printf("Error opening file!\n");
    return 1;  // Indicate failure
    }

//initiate the 1s in c1[][]
for(int i = 2*N/5; i<N/2; i++)
    for(int j=0; j<N/2; j++)
        c1[i][j] = 1;

MPI_Recv(parameter_double1, 4,MPI_DOUBLE,0,10,MPI_COMM_WORLD,&status);
double L, T, dt, dx;

L=parameter_double1[0];
T=parameter_double1[1];
dt=parameter_double1[2];
dx=parameter_double1[3];

//write initial data to file
for(int i = 0; i<N/2; i++)
    { 
    for(int j=0; j<N/2; j++)  
        fprintf(data1,"%.4f ",c1[i][j]);
    fprintf(data1,"\n");
    }

t3 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #0 and #3
    for(int j=0;j<N/2;j++)
        {
        send1to0[j]=c1[j][0];
        send1to0[N/2+j]=c1[j][N/2-1];

        send1to3[j]=c1[0][j];
        send1to3[N/2+j]=c1[N/2-1][j];
        }
    
    //MPI COMM with #0
    MPI_Send(send1to0,N,MPI_DOUBLE,0,n,MPI_COMM_WORLD);
    MPI_Recv(guard1from0, N,MPI_DOUBLE,0,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #3
    MPI_Recv(guard1from3, N,MPI_DOUBLE,3,n,MPI_COMM_WORLD,&status);
    MPI_Send(send1to3,N,MPI_DOUBLE,3,n,MPI_COMM_WORLD);

    //compute c1_new[0][1:N/2-1]. Need to use c3[N/2-1][*], which is stored  
    //in guard1from3[N/2+1:N-1]

    x= -L/2;
    v=(-1.414)*x;
    for(int j=1;j<N/2-1;j++)
    {
    y = -L/2.0 + (N/2 + j) * dx;
    u = 1.414 * y;
    //c1[-1][j] = guard1from3[N/2+j]
    c1_new[0][j]=.25*(guard1from3[N/2+j] + c1[1][j] + c1[0][j-1] + c1[0][j+1]) 
    - dt/(2*dx) * (u * (c1[1][j]-guard1from3[N/2+j]) + v * (c1[0][j+1]-c1[0][j-1]));
    }


    //compute c1_new[0][0], need to use c1[-1][0] = guard1from3[N/2] and c1[0][-1]=guard1from0[N/2]
    y = -L/2.0 + (N/2) * dx;
    u = 1.414 * y;
    //c1[0][-1]=guard1from0[N/2] 
    c1_new[0][0]=.25*(guard1from3[N/2] + c1[1][0] + guard1from0[N/2] + c1[0][1])
    - dt/(2*dx) * (u* (c1[1][0]-guard1from3[N/2]) + v* (c1[0][1]-guard1from0[N/2]));

    //compute c1_new[0][N/2-1], need to use c1[0][N/2]=guard1from0[0], c1[-1][N/2-1]=guard1from3[N-1]
    y = -L/2.0 + (N-1)*dx;
    u = 1.414 * y;
    c1_new[0][N/2-1]=.25*(guard1from3[N-1] + c1[1][N/2-1] + c1[0][N/2-2] + guard1from0[0])
    - dt/(2*dx) * ((u*(c1[1][N/2-1] - guard1from3[N-1])) + v* (guard1from0[0] - c1[0][N/2-2]));


    //compute c1_new from row 1 to row N/2-2
    #pragma omp parallel for default(none) shared(L,N,c1,c1_new,guard1from0,dx,dt) private(x,y,u,v) schedule(static) num_threads(nt)
    for(int i=1;i<N/2-1;i++)
       { 
        x= -L/2.0 + i*dx;
        v= -1.414 * x;
        //compute c1_new[i][0], c1[i][-1]=guard1from0[N/2+i], 
        y= -L/2.0 + (N/2) * dx;
        u=1.414*y;
        c1_new[i][0]=.25*(c1[i-1][0] + c1[i+1][0] + guard1from0[N/2+i] + c1[i][1]) + 
        - dt/(2*dx) * (u * (c1[i+1][0]-c1[i-1][0]) + v * (c1[i][1]-guard1from0[N/2+i]));
        for(int j=1;j<N/2-1;j++)
            {
            y = -L/2.0 + (N/2 + j)*dx;
            u = 1.414 * y;
            c1_new[i][j]=.25*(c1[i-1][j] + c1[i+1][j] + c1[i][j-1] + c1[i][j+1]) + 
            - dt/(2*dx) * (u * (c1[i+1][j]-c1[i-1][j]) + v * (c1[i][j+1]-c1[i][j-1]));
            }

        //c1[i][N/2] = guard1from0[i]
        y= -L/2.0 + (N-1)*dx;
        u=1.414*y;
        c1_new[i][N/2-1]=.25 * (c1[i-1][N/2-1] + c1[i+1][N/2-1] + c1[i][N/2-2] + guard1from0[i]) 
        - dt/(2*dx) * (u * (c1[i+1][N/2-1]-c1[i-1][N/2-1]) + v * (guard1from0[i]-c1[i][N/2-2]));

        }
    //compute c1_new[N/2-1][*]. Need to use c1[N/2][j] = guard1from3[j] for j = 1,2..N/2-2
    //special case c1[][]
        x = -L/2 + (N/2 - 1)*dx;
        v = -1.414 * x;
    
        for(int j=1; j<N/2-1; j++)
            {
            y = -L/2.0 + (N/2 + j)*dx;
            u = 1.414 * y;
            c1_new[N/2-1][j]=.25*(c1[N/2 - 2][j] + guard1from3[j] + c1[N/2 - 1][j-1] + c1[N/2 - 1][j+1])
            -dt/(2*dx) * (u * (guard1from3[j]-c1[N/2 - 2][j]) + v * (c1[N/2 - 1][j+1]-c1[N/2 - 1][j-1]));
            }

        y = -L/2.0 + (N/2)*dx;
        u = 1.414 * y;
        //c1[N/2-1][-1] = guard1from0[N-1]
        c1_new[N/2-1][0]=.25*(c1[N/2 - 2][0] + guard1from3[0] + guard1from0[N-1] + c1[N/2 - 1][1])
        -dt/(2*dx) * (u * (guard1from3[0]-c1[N/2 - 2][0]) + v * (c1[N/2 - 1][1]-guard1from0[N-1]));
        

        //c1[N/2-1][N/2] = guard1from0[N/2-1], c1[N/2][N/2-1] = guard1from3[N/2-1]
        y = -L/2.0 + (N-1)*dx;
        u = 1.414 * y;
        c1_new[N/2-1][N/2-1]=.25*(c1[N/2 - 2][N/2-1] + guard1from3[N/2-1] + c1[N/2-1][N/2-2] + guard1from0[N/2-1])
        -dt/(2*dx) * (u * (guard1from3[N/2-1]-c1[N/2 - 2][N/2-1]) + v * (guard1from0[N/2-1]-c1[N/2-1][N/2-2]));
        
        
        if(n==NT/2||n==NT)
        {
            for(int i=0;i<N/2;i++)
                {
                for(int j=0;j<N/2;j++)
                    fprintf(data1,"%.4f ",c1_new[i][j]); 
                fprintf(data1,"\n"); 
                }
        } 
    
    pt1=c1;
    c1=c1_new;
    c1_new=pt1;
    }    
t4 = omp_get_wtime();
fclose(data1);

}

else if(mype==2)
{
FILE *data2=fopen("lax2.txt", "w");
if(data2==NULL)
    {
    printf("Error opening file!\n");
    return 1;  // Indicate failure
    }
MPI_Recv(parameter_double2, 4,MPI_DOUBLE,0,10,MPI_COMM_WORLD,&status);
double L, T, dt, dx;

L=parameter_double2[0];
T=parameter_double2[1];
dt=parameter_double2[2];
dx=parameter_double2[3];
//initiate the 1s in c2[][]
for(int i = 0; i<N/10; i++)
    for(int j=0; j<N/2; j++)
        c2[i][j] = 1;

//write initial data to file
for(int i = 0; i<N/2; i++)
    { 
    for(int j=0; j<N/2; j++)  
        fprintf(data2,"%.4f ",c2[i][j]);
    fprintf(data2,"\n");
    }

t5 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #0 and #3
    for(int j=0;j<N/2;j++)
        {
        send2to0[j]=c2[0][j];
        send2to0[N/2+j]=c2[N/2-1][j];

        send2to3[j]=c2[j][0];
        send2to3[N/2+j]=c2[j][N/2-1];
        }
    
    //MPI COMM with #0
    MPI_Send(send2to0,N,MPI_DOUBLE,0,n,MPI_COMM_WORLD);
    MPI_Recv(guard2from0, N,MPI_DOUBLE,0,n,MPI_COMM_WORLD,&status);


    //MPI COMM with #3
    MPI_Recv(guard2from3, N,MPI_DOUBLE,3,n,MPI_COMM_WORLD,&status);
    MPI_Send(send2to3,N,MPI_DOUBLE,3,n,MPI_COMM_WORLD);

    //compute c2_new[0][j]. for j=1,2,..,N/2-2  
    //c2[-1][j]=guard2from0[N/2+j]
    x= -L/2.0 + (N/2)*dx;
    v= (-1.414)*x;
    for(int j=1;j<N/2-1;j++)
        {
        y = -L/2.0 + j*dx;
        u = 1.414 * y;
        c2_new[0][j]=.25*(guard2from0[N/2+j] + c2[1][j] + c2[0][j-1] + c2[0][j+1]) 
        - dt/(2*dx) * (u * (c2[1][j]-guard2from0[N/2+j]) + v * (c2[0][j+1]-c2[0][j-1]));
        }
    
    //compute c2_new[0][0], need to use c2[-1][0] = guard2from0[N/2] and c2[0][-1]=guard2from3[N/2]
    y = -L/2.0;
    u = 1.414 * y;
    c2_new[0][0]=.25*(guard2from0[N/2] + c2[1][0] + c2[0][1] + guard2from3[N/2])
    - dt/(2*dx) * (u* (c2[1][0]-guard2from0[N/2]) + v* (guard2from3[N/2]-c2[0][1]));

    //compute c2_new[0][N/2-1], need to use c2[0][N/2]=guard2from3[0],c2[-1][N/2-1]=guard2from0[N-1]
    y = -L/2.0 + dx*(N/2-1);
    u = 1.414 * y;
    c2_new[0][N/2-1]=.25*(guard2from0[N-1] + c2[1][N/2-1] + c2[0][N/2-2] + guard2from3[0])
    - dt/(2*dx) * ((u*(c2[1][N/2-1] - guard2from0[N-1])) + v* (guard2from3[0] - c2[0][N/2-2]));


    //compute c2_new from row 1 to row N/2-2
    #pragma omp parallel for default(none) shared(L,N,c2,c2_new,guard2from3,dx,dt) private(x,y,u,v) schedule(static) num_threads(nt)
    for(int i=1;i<N/2-1;i++)
       { 
        x= -L/2.0 + (N/2+i)*dx;
        v= -1.414 * x;
        //compute c2_new[i][0], c2[i][-1]=guard2from3[N/2+i]
        y = -L/2.0;
        u = 1.414 * y;
        c2_new[i][0]=.25*(c2[i-1][0] + c2[i+1][0] + guard2from3[N/2+i] + c2[i][1])  
        - dt/(2*dx) * (u * (c2[i+1][0]-c2[i-1][0]) + v * (c2[i][1]-guard2from3[N/2+i]));
        //compute c2_new[i][j] for j=1,2,...N/2-2
        for(int j=1;j<N/2-1;j++)
            {
            y = -L/2.0 + j*dx;
            u = 1.414 * y;
            c2_new[i][j]=.25*(c2[i-1][j] + c2[i+1][j] + c2[i][j-1] + c2[i][j+1]) 
            - dt/(2*dx) * (u * (c2[i+1][j]-c2[i-1][j]) + v * (c2[i][j+1]-c2[i][j-1]));
            }
        //compute c2_new[i][N/2-1], c2[i][N/2]=guard2from3[i]
        y = -L/2.0 + dx * (N/2 - 1);
        u = 1.414 * y;
        c2_new[i][N/2-1]=.25*(c2[i-1][N/2-1] + c2[i+1][N/2-1] + c2[i][N/2-2] + guard2from3[i]) 
        - dt/(2*dx) * (u * (c2[i+1][N/2-1]-c2[i-1][N/2-1]) + v * (guard2from3[i]-c2[i][N/2-2]));
        }

    //compute c2_new[N/2-1][j] for j = 1,2..N/2-2.   Need to use c2[N/2][j] = guard2from0[j] 
    //
        x = -L/2.0 + (N-1)*dx;
        v = -1.414 * x;
        for(int j=1; j<N/2-1; j++)
            {
            y = -L/2.0 + j*dx;
            u = 1.414 * y;
            c2_new[N/2-1][j]=.25*(c2[N/2 - 2][j] + guard2from0[j] + c2[N/2 - 1][j-1] + c2[N/2 - 1][j+1])
            -dt/(2*dx) * (u * (guard2from0[j]-c2[N/2 - 2][j]) + v * (c2[N/2 - 1][j+1]-c2[N/2 - 1][j-1]));
            }

        //compute c2_new[N/2-1][0], need to use c2[N/2][0] = guard2from0[0] 
        //and c2[N/2-1][-1]=guard2from3[N-1]   
        y = -L/2.0;
        u = 1.414 * y;
        c2_new[N/2-1][0]=.25*(c2[N/2 - 2][0] + guard2from0[0] + guard2from3[N-1] + c2[N/2 - 1][1])
        -dt/(2*dx) * (u * (guard2from0[0]-c2[N/2 - 2][0]) + v * (c2[N/2 - 1][1]-guard2from3[N-1]));
        

        //compute c2_new[N/2-1][N/2-1], need to use c2[N/2-1][N/2] = guard2from3[N/2-1]
        //and c2[N/2][N/2-1]= guard2from0[N/2-1] 
        y = -L/2.0 + (N/2-1)*dx;
        u = 1.414 * y;
        c2_new[N/2-1][N/2-1]=.25*(c2[N/2 - 2][N/2-1] + guard2from0[N/2-1] + c2[N/2-1][N/2-2] + guard2from3[N/2-1])
        -dt/(2*dx) * (u * (guard2from0[N/2-1]-c2[N/2 - 2][N/2-1]) + v * (guard2from3[N/2-1]-c2[N/2-1][N/2-2]));
        
    
    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/2;i++)
            {
            for(int j=0;j<N/2;j++)
                fprintf(data2,"%.4f ",c2_new[i][j]); 
            fprintf(data2,"\n"); 
            }
       } 
    pt2=c2;
    c2=c2_new;
    c2_new=pt2;
}
t6 = omp_get_wtime();
fclose(data2);
}



else if(mype==3)
{
FILE *data3=fopen("lax3.txt", "w");
if(data3==NULL)
    {
    printf("Error opening file!\n");
    return 1;  // Indicate failure
    }
MPI_Recv(parameter_double3, 4,MPI_DOUBLE,0,10,MPI_COMM_WORLD,&status);
double L, T, dt, dx;

L=parameter_double3[0];
T=parameter_double3[1];
dt=parameter_double3[2];
dx=parameter_double3[3];
//initiate the 1s in c3[][]
for(int i = 0; i<N/10; i++)
    for(int j=0; j<N/2; j++)
        c3[i][j] = 1;


//write initial data to file
for(int i = 0; i<N/2; i++)
    { 
    for(int j=0; j<N/2; j++)  
        fprintf(data3,"%.4f ",c3[i][j]);
    fprintf(data3,"\n");
    }

t7 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #1 and #2
    for(int j=0;j<N/2;j++)
        {
        send3to1[j]=c3[0][j];
        send3to1[N/2+j]=c3[N/2-1][j];
        
        send3to2[j]=c3[j][0];
        send3to2[N/2+j]=c3[j][N/2-1];
        }

    //MPI COMM with #1
    MPI_Send(send3to1,N,MPI_DOUBLE,1,n,MPI_COMM_WORLD);
    MPI_Recv(guard3from1, N,MPI_DOUBLE,1,n,MPI_COMM_WORLD,&status);


    //MPI COMM with #2
    MPI_Send(send3to2,N,MPI_DOUBLE,2,n,MPI_COMM_WORLD);
    MPI_Recv(guard3from2, N,MPI_DOUBLE,2,n,MPI_COMM_WORLD,&status);

    //compute c3_new[0][j], for j= 1,2,..,N/2-2. Need to use c3[-1][j]=guard3from1[N/2 + j]
    x= -L/2.0 + (N/2) * dx;
    v= (-1.414)*x;
    for(int j=1;j<N/2-1;j++)
        {
        y = -L/2.0 + (N/2 + j)*dx;
        u = 1.414 * y;
        c3_new[0][j]=.25*(guard3from1[N/2+j] + c3[1][j] + c3[0][j-1] + c3[0][j+1]) 
        - dt/(2*dx) * (u * (c3[1][j]-guard3from1[N/2+j]) + v * (c3[0][j+1]-c3[0][j-1]));
        }
    
    //compute c3_new[0][0], need to use c3[-1][0] = guard3from1[N/2] and c3[0][-1]=guard3from2[N/2]
    y = -L/2.0 + (N/2)*dx;
    u = 1.414 * y;
    c3_new[0][0]=.25*(guard3from1[N/2] + c3[1][0] + guard3from2[N/2]+ c3[0][1])
    - dt/(2*dx) * (u* (c3[1][0]-guard3from1[N/2]) + v* (c3[0][1]-guard3from2[N/2]));

    //compute c3_new[0][N/2-1], need to use c3[0][N/2]=guard3from2[0],
    //and c3[-1][N/2-1]=guard3from1[N-1]
    y = -L/2.0 + dx*(N-1) ;
    u = 1.414 * y;
    c3_new[0][N/2-1]=.25*(guard3from1[N-1] + c3[1][N/2-1] + c3[0][N/2-2] + guard3from2[0])
    - dt/(2*dx) * ((u*(c3[1][N/2-1] - guard3from1[N-1])) + v* (guard3from2[0] - c3[0][N/2-2]));


    //compute c3_new from row 1 to row N/2-2
    #pragma omp parallel for default(none) shared(L,N,c3,c3_new,guard3from2,dx,dt) private(x,y,u,v) schedule(static) num_threads(nt)
    for(int i=1;i<N/2-1;i++)
       { 
        x= -L/2.0 + (N/2 + i) * dx;
        v= -1.414 * x;
        //compute c3_new[i][0], c3[i][-1]=guard3from2[N/2+i]
        y = -L/2.0 + (N/2) *dx;
        u = 1.414 * y;
        c3_new[i][0]=.25*(c3[i-1][0] + c3[i+1][0] + guard3from2[N/2+i] + c3[i][1])  
        - dt/(2*dx) * (u * (c3[i+1][0]-c3[i-1][0]) + v * (c3[i][1]-guard3from2[N/2+i]));
        //compute c3_new[i][j] for j=1,2,...N/2-2
        for(int j=1;j<N/2-1;j++)
            {
            y = -L/2.0 + L/2.0 + j*dx;
            u = 1.414 * y;
            c3_new[i][j]=.25*(c3[i-1][j] + c3[i+1][j] + c3[i][j-1] + c3[i][j+1]) 
            - dt/(2*dx) * (u * (c3[i+1][j]-c3[i-1][j]) + v * (c3[i][j+1]-c3[i][j-1]));
            }
        //compute c3_new[i][N/2-1], c3[i][N/2]=guard3from2[i]
        y = -L/2.0 + dx*(N-1);
        u = 1.414 * y;
        c3_new[i][N/2-1]=.25*(c3[i-1][N/2-1] + c3[i+1][N/2-1] + c3[i][N/2-2] + guard3from2[i]) 
        - dt/(2*dx) * (u * (c3[i+1][N/2-1]-c3[i-1][N/2-1]) + v * (guard3from2[i]-c3[i][N/2-2]));
        }

        //compute c3_new[N/2-1][j]. Need to use c3[N/2][j] = guard3from1[j] for j = 1,2..N/2-2
        x = -L/2.0 + (N-1)*dx ;
        v = -1.414 * x;
        for(int j=1; j<N/2-1; j++)
            {
            y = -L/2.0 + (N/2+ j)*dx;
            u = 1.414 * y;
            c3_new[N/2-1][j]=.25*(c3[N/2 - 2][j] + guard3from1[j] + c3[N/2 - 1][j-1] + c3[N/2 - 1][j+1])
            -dt/(2*dx) * (u * (guard3from1[j]-c3[N/2 - 2][j]) + v * (c3[N/2 - 1][j+1]-c3[N/2 - 1][j-1]));
            }

        
        //compute c3_new[N/2-1][0], need to use c3[N/2][0] = guard3from1[0] 
        //and c3[N/2-1][-1]=guard3from2[N-1]   
        y = -L/2.0 +(N/2)*dx ;
        u = 1.414 * y;
        c3_new[N/2-1][0]=.25*(c3[N/2 - 2][0] + guard3from1[0] + guard3from2[N-1] + c3[N/2 - 1][1])
        -dt/(2*dx) * (u * (guard3from1[0]-c3[N/2 - 2][0]) + v * (c3[N/2 - 1][1]-guard3from2[N-1]));
        

        //compute c3_new[N/2-1][N/2-1], need to use c3[N/2-1][N/2] = guard3from2[N/2-1]
        //and c3[N/2][N/2-1]= guard3from1[N/2-1] 
        y = -L/2.0 + (N-1)*dx;
        u = 1.414 * y;
        c3_new[N/2-1][N/2-1]=.25*(c3[N/2 - 2][N/2-1] + guard3from1[N/2-1] + c3[N/2-1][N/2-2] + guard3from2[N/2-1])
        -dt/(2*dx) * (u * (guard3from1[N/2-1]-c3[N/2 - 2][N/2-1]) + v * (guard3from2[N/2-1]-c3[N/2-1][N/2-2]));

    
    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/2;i++)
            {
            for(int j=0;j<N/2;j++)
                fprintf(data3,"%.4f ",c3_new[i][j]); 
            fprintf(data3,"\n"); 
            }
       } 
    pt3=c3;
    c3=c3_new;
    c3_new=pt3;
}
t8 = omp_get_wtime();
fclose(data3);
}

MPI_Finalize();
double Nd=N, NTd=NT;


if(mype==0)
printf("%d, Time used: %f, grind_rate: %.2fM\n", mype, t2-t1, Nd*Nd*NTd/((t2-t1)*1000000));

if(mype==1)
printf("%d, Time used: %f, grind_rate: %.2fM\n", mype, t4-t3, Nd*Nd*NTd/((t4-t3)*1000000));

if(mype==2)
printf("%d, Time used: %f, grind_rate: %.2fM\n", mype, t6-t5, Nd*Nd*NTd/((t6-t5)*1000000));

if(mype==3)
printf("%d, Time used: %f, grind_rate: %.2fM\n", mype, t8-t7, Nd*Nd*NTd/((t8-t7)*1000000));
/*

free(c0[0]);
free(c1[0]);
free(c2[0]);
free(c3[0]);
free(c0_new[0]);
free(c1_new[0]);
free(c2_new[0]);
free(c3_new[0]);
free(c0);
free(c1);
free(c2);
free(c3);
free(c0_new);
free(c1_new);
free(c2_new);
free(c3_new);

free(guard0from1);
free(guard0from2);
free(send0to1);
free(send0to2);

free(guard1from0);
free(guard1from3);
free(send1to0);
free(send1to3);

free(guard2from0);
free(guard2from3);
free(send2to0);
free(send2to3);

free(guard3from1);
free(guard3from2);
free(send3to1);
free(send3to2);
*/
return 0;
}