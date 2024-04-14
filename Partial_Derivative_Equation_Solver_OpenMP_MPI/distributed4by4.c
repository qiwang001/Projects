#include "stdio.h"
#include "stdlib.h"
#include "assert.h"
#include "mpi.h"
#include"omp.h"
#include"math.h"
//export OMPI_CC=gcc-13
double**
datamatrix(int n)
{
    double *data = (double*)calloc(n * n ,  sizeof(double));
    double **M = (double**)malloc(n * sizeof(double*)); 
    for (unsigned long i=0; i < n; i++)
        M[i] = data + n * i;
        //M[i] = &data[n * i];
    return M;
}

int gce()
{

    return 0;
}

int
main(int argc, char** argv)
{
int stat, nprocs, mype,im1,ia1,jm1,ja1,NT;
double L=1,T=1, dt=1.25e-4, dx,x,y,u,v;
double t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,
t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,t31,t32;
int nt=atoi(argv[2]), N=atoi(argv[1]);
double **pt0,**pt1,**pt2, **pt3,**pt4, **pt5,**pt6,**pt7, **pt8,
**pt9,**pt10,**pt11, **pt12,**pt13,**pt14,**pt15;
double **c0 = datamatrix(N/4);
double **c0_new = datamatrix(N/4);
double **c1 = datamatrix(N/4);
double **c1_new = datamatrix(N/4);
double **c2 = datamatrix(N/4);
double **c2_new = datamatrix(N/4);
double **c3 = datamatrix(N/4);
double **c3_new = datamatrix(N/4);
double **c4 = datamatrix(N/4);
double **c4_new = datamatrix(N/4);
double **c5 = datamatrix(N/4);
double **c5_new = datamatrix(N/4);
double **c6 = datamatrix(N/4);
double **c6_new = datamatrix(N/4);
double **c7 = datamatrix(N/4);
double **c7_new = datamatrix(N/4);
double **c8 = datamatrix(N/4);
double **c8_new = datamatrix(N/4);
double **c9 = datamatrix(N/4);
double **c9_new = datamatrix(N/4);
double **c10 = datamatrix(N/4);
double **c10_new = datamatrix(N/4);
double **c11 = datamatrix(N/4);
double **c11_new = datamatrix(N/4);
double **c12 = datamatrix(N/4);
double **c12_new = datamatrix(N/4);
double **c13 = datamatrix(N/4);
double **c13_new = datamatrix(N/4);
double **c14 = datamatrix(N/4);
double **c14_new = datamatrix(N/4);
double **c15 = datamatrix(N/4);
double **c15_new = datamatrix(N/4);

double* guard0from1 = (double*)malloc(sizeof(double)*N/4);
double* guard0from3 = (double*)malloc(sizeof(double)*N/4);
double* guard0from4 = (double*)malloc(sizeof(double)*N/4);
double* guard0from12 = (double*)malloc(sizeof(double)*N/4);
double* send0to1 = (double*)malloc(sizeof(double)*N/4);
double* send0to3 = (double*)malloc(sizeof(double)*N/4);
double* send0to4 = (double*)malloc(sizeof(double)*N/4);
double* send0to12 = (double*)malloc(sizeof(double)*N/4);

double* guard1from0 = (double*)malloc(sizeof(double)*N/4);
double* guard1from2 = (double*)malloc(sizeof(double)*N/4);
double* guard1from5 = (double*)malloc(sizeof(double)*N/4);
double* guard1from13 = (double*)malloc(sizeof(double)*N/4);
double* send1to0 = (double*)malloc(sizeof(double)*N/4);
double* send1to2 = (double*)malloc(sizeof(double)*N/4);
double* send1to5 = (double*)malloc(sizeof(double)*N/4);
double* send1to13 = (double*)malloc(sizeof(double)*N/4);

double* guard2from1 = (double*)malloc(sizeof(double)*N/4);
double* guard2from3 = (double*)malloc(sizeof(double)*N/4);
double* guard2from6 = (double*)malloc(sizeof(double)*N/4);
double* guard2from14 = (double*)malloc(sizeof(double)*N/4);
double* send2to1 = (double*)malloc(sizeof(double)*N/4);
double* send2to3 = (double*)malloc(sizeof(double)*N/4);
double* send2to6 = (double*)malloc(sizeof(double)*N/4);
double* send2to14 = (double*)malloc(sizeof(double)*N/4);

double* guard3from0 = (double*)malloc(sizeof(double)*N/4);
double* guard3from2 = (double*)malloc(sizeof(double)*N/4);
double* guard3from7 = (double*)malloc(sizeof(double)*N/4);
double* guard3from15 = (double*)malloc(sizeof(double)*N/4);
double* send3to0 = (double*)malloc(sizeof(double)*N/4);
double* send3to2 = (double*)malloc(sizeof(double)*N/4);
double* send3to7 = (double*)malloc(sizeof(double)*N/4);
double* send3to15 = (double*)malloc(sizeof(double)*N/4);

double* guard4from0 = (double*)malloc(sizeof(double)*N/4);
double* guard4from5 = (double*)malloc(sizeof(double)*N/4);
double* guard4from7 = (double*)malloc(sizeof(double)*N/4);
double* guard4from8 = (double*)malloc(sizeof(double)*N/4);
double* send4to0 = (double*)malloc(sizeof(double)*N/4);
double* send4to5 = (double*)malloc(sizeof(double)*N/4);
double* send4to7 = (double*)malloc(sizeof(double)*N/4);
double* send4to8 = (double*)malloc(sizeof(double)*N/4);

double* guard5from1 = (double*)malloc(sizeof(double)*N/4);
double* guard5from4 = (double*)malloc(sizeof(double)*N/4);
double* guard5from6 = (double*)malloc(sizeof(double)*N/4);
double* guard5from9 = (double*)malloc(sizeof(double)*N/4);
double* send5to1 = (double*)malloc(sizeof(double)*N/4);
double* send5to4 = (double*)malloc(sizeof(double)*N/4);
double* send5to6 = (double*)malloc(sizeof(double)*N/4);
double* send5to9 = (double*)malloc(sizeof(double)*N/4);

double* guard6from2 = (double*)malloc(sizeof(double)*N/4);
double* guard6from5 = (double*)malloc(sizeof(double)*N/4);
double* guard6from7 = (double*)malloc(sizeof(double)*N/4);
double* guard6from10 = (double*)malloc(sizeof(double)*N/4);
double* send6to2 = (double*)malloc(sizeof(double)*N/4);
double* send6to5 = (double*)malloc(sizeof(double)*N/4);
double* send6to7 = (double*)malloc(sizeof(double)*N/4);
double* send6to10 = (double*)malloc(sizeof(double)*N/4);


double* guard7from3 = (double*)malloc(sizeof(double)*N/4);
double* guard7from4 = (double*)malloc(sizeof(double)*N/4);
double* guard7from6 = (double*)malloc(sizeof(double)*N/4);
double* guard7from11 = (double*)malloc(sizeof(double)*N/4);
double* send7to3 = (double*)malloc(sizeof(double)*N/4);
double* send7to4 = (double*)malloc(sizeof(double)*N/4);
double* send7to6 = (double*)malloc(sizeof(double)*N/4);
double* send7to11 = (double*)malloc(sizeof(double)*N/4);

double* guard8from4 = (double*)malloc(sizeof(double)*N/4);
double* guard8from9 = (double*)malloc(sizeof(double)*N/4);
double* guard8from11 = (double*)malloc(sizeof(double)*N/4);
double* guard8from12 = (double*)malloc(sizeof(double)*N/4);
double* send8to4 = (double*)malloc(sizeof(double)*N/4);
double* send8to9 = (double*)malloc(sizeof(double)*N/4);
double* send8to11 = (double*)malloc(sizeof(double)*N/4);
double* send8to12 = (double*)malloc(sizeof(double)*N/4);

double* guard9from5 = (double*)malloc(sizeof(double)*N/4);
double* guard9from8 = (double*)malloc(sizeof(double)*N/4);
double* guard9from10 = (double*)malloc(sizeof(double)*N/4);
double* guard9from13 = (double*)malloc(sizeof(double)*N/4);
double* send9to5 = (double*)malloc(sizeof(double)*N/4);
double* send9to8 = (double*)malloc(sizeof(double)*N/4);
double* send9to10 = (double*)malloc(sizeof(double)*N/4);
double* send9to13 = (double*)malloc(sizeof(double)*N/4);

double* guard10from6 = (double*)malloc(sizeof(double)*N/4);
double* guard10from9 = (double*)malloc(sizeof(double)*N/4);
double* guard10from11 = (double*)malloc(sizeof(double)*N/4);
double* guard10from14 = (double*)malloc(sizeof(double)*N/4);
double* send10to6 = (double*)malloc(sizeof(double)*N/4);
double* send10to9 = (double*)malloc(sizeof(double)*N/4);
double* send10to11 = (double*)malloc(sizeof(double)*N/4);
double* send10to14 = (double*)malloc(sizeof(double)*N/4);

double* guard11from7 = (double*)malloc(sizeof(double)*N/4);
double* guard11from8 = (double*)malloc(sizeof(double)*N/4);
double* guard11from10 = (double*)malloc(sizeof(double)*N/4);
double* guard11from15 = (double*)malloc(sizeof(double)*N/4);
double* send11to7 = (double*)malloc(sizeof(double)*N/4);
double* send11to8 = (double*)malloc(sizeof(double)*N/4);
double* send11to10 = (double*)malloc(sizeof(double)*N/4);
double* send11to15 = (double*)malloc(sizeof(double)*N/4);

double* guard12from0 = (double*)malloc(sizeof(double)*N/4);
double* guard12from8 = (double*)malloc(sizeof(double)*N/4);
double* guard12from13 = (double*)malloc(sizeof(double)*N/4);
double* guard12from15 = (double*)malloc(sizeof(double)*N/4);
double* send12to0 = (double*)malloc(sizeof(double)*N/4);
double* send12to8 = (double*)malloc(sizeof(double)*N/4);
double* send12to13 = (double*)malloc(sizeof(double)*N/4);
double* send12to15 = (double*)malloc(sizeof(double)*N/4);

double* guard13from1 = (double*)malloc(sizeof(double)*N/4);
double* guard13from9 = (double*)malloc(sizeof(double)*N/4);
double* guard13from12 = (double*)malloc(sizeof(double)*N/4);
double* guard13from14 = (double*)malloc(sizeof(double)*N/4);
double* send13to1 = (double*)malloc(sizeof(double)*N/4);
double* send13to9 = (double*)malloc(sizeof(double)*N/4);
double* send13to12 = (double*)malloc(sizeof(double)*N/4);
double* send13to14 = (double*)malloc(sizeof(double)*N/4);


double* guard14from2 = (double*)malloc(sizeof(double)*N/4);
double* guard14from10 = (double*)malloc(sizeof(double)*N/4);
double* guard14from13 = (double*)malloc(sizeof(double)*N/4);
double* guard14from15 = (double*)malloc(sizeof(double)*N/4);
double* send14to2 = (double*)malloc(sizeof(double)*N/4);
double* send14to10 = (double*)malloc(sizeof(double)*N/4);
double* send14to13 = (double*)malloc(sizeof(double)*N/4);
double* send14to15 = (double*)malloc(sizeof(double)*N/4);

double* guard15from3 = (double*)malloc(sizeof(double)*N/4);
double* guard15from11 = (double*)malloc(sizeof(double)*N/4);
double* guard15from12 = (double*)malloc(sizeof(double)*N/4);
double* guard15from14 = (double*)malloc(sizeof(double)*N/4);
double* send15to3 = (double*)malloc(sizeof(double)*N/4);
double* send15to11 = (double*)malloc(sizeof(double)*N/4);
double* send15to12 = (double*)malloc(sizeof(double)*N/4);
double* send15to14 = (double*)malloc(sizeof(double)*N/4);

NT = 1/dt;
dx=L/(N-1);
assert(dt<=(dx/2));
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

//write initial data to file
for(int i = 0; i<N/4; i++)
    { 
    for(int j=0; j<N/4; j++)  
        fprintf(data0,"%.4f ",c0[i][j]);
    fprintf(data0,"\n");
    }

t1 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {

    if(n%100==0)
        {
        printf("NT is %d, going well on %d th iteration..\n", NT, n);
        printf("time eclipsed:%f \n", omp_get_wtime()-t1);
        }
    //prepare data to send to process #1 and #2
    for(int j=0;j<N/4;j++)
        {
        send0to1[j]=c0[j][N/4-1];
        send0to3[j]=c0[j][0];

        send0to4[j]=c0[N/4-1][j];
        send0to12[j]=c0[0][j];
        }

    //MPI COMM with #1
    MPI_Recv(guard0from1, N/4,MPI_DOUBLE,1,n,MPI_COMM_WORLD,&status);
    MPI_Send(send0to1,N/4,MPI_DOUBLE,1,n,MPI_COMM_WORLD);

    //MPI COMM with #3
    MPI_Recv(guard0from3, N/4,MPI_DOUBLE,3,n,MPI_COMM_WORLD,&status);
    MPI_Send(send0to3,N/4,MPI_DOUBLE,3,n,MPI_COMM_WORLD);

    //MPI COMM with #4
    MPI_Recv(guard0from4, N/4,MPI_DOUBLE,4,n,MPI_COMM_WORLD,&status);
    MPI_Send(send0to4,N/4,MPI_DOUBLE,4,n,MPI_COMM_WORLD);

    //MPI COMM with #12
    MPI_Recv(guard0from12, N/4,MPI_DOUBLE,12,n,MPI_COMM_WORLD,&status);
    MPI_Send(send0to12,N/4,MPI_DOUBLE,12,n,MPI_COMM_WORLD);


    //compute c0_new[0][j].for j=1,2,..,N/2-2, c2[N/2-1][*]
    //c0[-1][j] = guard0from2[N/2 + j]
    x= -L/2;
    v=(-1.414)*x;
    for(int j=1;j<N/4-1;j++)
    {
    y = -L/2.0 + j*dx;
    u = 1.414 * y;
    c0_new[0][j]=.25*(guard0from12[j] + c0[1][j] + c0[0][j-1] + c0[0][j+1]) 
    - dt/(2*dx) * (u * (c0[1][j]-guard0from12[j]) + v * (c0[0][j+1]-c0[0][j-1]));
    }

    //compute c0_new[0][0] , need to use c[-1][0] = guard0from2[N/2] and c[0][-1]=guard0from1[N/2]
    y = -L/2.0;
    u = 1.414 * y;
    c0_new[0][0]=.25*(guard0from12[0] + c0[1][0] + guard0from3[0] + c0[0][1])
    - dt/(2*dx) * (u* (c0[1][0]-guard0from12[0]) + v* (c0[0][1]-guard0from3[0]));

    //compute c0_new[0][N/4-1], need to use c[0][N/2]=guard0from1[0], c[-1][N/2-1]=guard0from2[N-1]
    y = -L/2.0 + dx * (N/4-1);
    u = 1.414 * y;
    c0_new[0][N/4-1]=.25*(guard0from12[N/4-1] + c0[1][N/4-1] + c0[0][N/4-2] + guard0from1[0])
    - dt/(2*dx) * ((u*(c0[1][N/4-1] - guard0from12[N/4-1])) + v* (guard0from1[0] - c0[0][N/4-2]));

    //compute c0_new from row 1 to row N/4-2
    #pragma omp parallel for default(shared) schedule(static) num_threads(nt)
    for(int i=1;i<N/4-1;i++)
       { 
        x= -L/2.0 + i*dx;
        v= -1.414 * x;

        //compute c_new[i][0]
        y= -L/2; 
        u= 1.414*y;
        c0_new[i][0]=.25*(c0[i-1][0] + c0[i+1][0] + guard0from3[i] + c0[i][1])
        - dt/(2*dx) * (u * (c0[i+1][0]-c0[i-1][0]) + v * (c0[i][1]-guard0from3[i]));
        //compute c0_new[i][j] for j=1,2..,N/4-2.
        for(int j=1;j<N/4-1;j++)
            {
            y = -L/2.0 + j*dx;
            u = 1.414 * y;
            c0_new[i][j]=.25*(c0[i-1][j] + c0[i+1][j] + c0[i][j-1] + c0[i][j+1]) 
            - dt/(2*dx) * (u * (c0[i+1][j]-c0[i-1][j]) + v * (c0[i][j+1]-c0[i][j-1]));
            }

        //compute c0_new[i][N/4-1]
        y=-L/2 + (N/4-1)*dx;
        u = 1.414*y;
        c0_new[i][N/4-1]=.25*(c0[i-1][N/4-1] + c0[i+1][N/4-1] + c0[i][N/4-2] + guard0from1[i]) 
        - dt/(2*dx) * (u * (c0[i+1][N/4-1]-c0[i-1][N/4-1]) + v * (guard0from1[i]-c0[i][N/4-2]));

        }

        //compute c0_new[N/4-1][*]. Need to use c[N/2][j] = guard0from2[j] for j = 1,2..N/2-2
        x = -L/2 + (N/4 - 1)*dx;
        v = -1.414 * x;
        for(int j=1; j<N/4-1; j++)
            {
            y = -L/2.0 + j*dx;
            u = 1.414 * y;
            c0_new[N/4-1][j]=.25*(c0[N/4 - 2][j] + guard0from4[j] + c0[N/4 - 1][j-1] + c0[N/4 - 1][j+1])
            -dt/(2*dx) * (u * (guard0from4[j]-c0[N/4 - 2][j]) + v * (c0[N/4 - 1][j+1]-c0[N/4 - 1][j-1]));
            }
        //compute c0_new[N/4-1][0], c0[N/2][0]=guard0from2[0], c0[N/2-1][-1]=guard0from1[N-1]
        y = -L/2.0;
        u = 1.414 * y;
        c0_new[N/4-1][0]=.25*(c0[N/4 - 2][0] + guard0from4[0] + guard0from3[N/4-1] + c0[N/4 - 1][1])
        -dt/(2*dx) * (u * (guard0from4[0]-c0[N/4 - 2][0]) + v * (c0[N/4 - 1][1]-guard0from3[N/4-1]));

        //c0[N/2-1][N/2] = guard0from1[N/2-1], c0[N/2][N/2-1]=guard0from2[N/2-1]
        y = -L/2.0 + (N/4-1)*dx;
        u = 1.414 * y;
        c0_new[N/4-1][N/4-1]=.25*(c0[N/4 - 2][N/4-1] + guard0from4[N/4-1] + c0[N/4-1][N/4-2] + guard0from1[N/4-1])
        -dt/(2*dx) * (u * (guard0from4[N/4-1]-c0[N/4 - 2][N/4-1]) + v * (guard0from1[N/4-1]-c0[N/4-1][N/4-2]));

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

//write initial data to file
for(int i = 0; i<N/4; i++)
    { 
    for(int j=0; j<N/4; j++)  
        fprintf(data1,"%.4f ",c1[i][j]);
    fprintf(data1,"\n");
    }

t3 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #1 and #2
    for(int j=0;j<N/4;j++)
        {
        send1to0[j]=c1[j][0];
        send1to2[j]=c1[j][N/4-1];
        send1to5[j]=c1[N/4-1][j];
        send1to13[j]=c1[0][j];
        }

    //MPI COMM with #0
    MPI_Send(send1to0,N/4,MPI_DOUBLE,0,n,MPI_COMM_WORLD);
    MPI_Recv(guard1from0, N/4,MPI_DOUBLE,0,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #2
    MPI_Recv(guard1from2, N/4,MPI_DOUBLE,2,n,MPI_COMM_WORLD,&status);
    MPI_Send(send1to2,N/4,MPI_DOUBLE,2,n,MPI_COMM_WORLD);

    //MPI COMM with #5
    MPI_Recv(guard1from5, N/4,MPI_DOUBLE,5,n,MPI_COMM_WORLD,&status);
    MPI_Send(send1to5,N/4,MPI_DOUBLE,5,n,MPI_COMM_WORLD);

    //MPI COMM with #13
    MPI_Recv(guard1from13, N/4,MPI_DOUBLE,13,n,MPI_COMM_WORLD,&status);
    MPI_Send(send1to13,N/4,MPI_DOUBLE,13,n,MPI_COMM_WORLD);


    //compute c1_new[0][j].for j=1,2,..,N/2-2, c2[N/2-1][*]
    //c0[-1][j] = guard0from2[N/2 + j]
    x= -L/2;
    v=(-1.414)*x;
    for(int j=1;j<N/4-1;j++)
        {
        y = -L/2.0 + (N/4+j)*dx;
        u = 1.414 * y;
        c1_new[0][j]=.25*(guard1from13[j] + c1[1][j] + c1[0][j-1] + c1[0][j+1]) 
        - dt/(2*dx) * (u * (c1[1][j]-guard1from13[j]) + v * (c1[0][j+1]-c1[0][j-1]));
        }

    //compute c1_new[0][0] , need to use c[-1][0] = guard0from2[N/2] and c[0][-1]=guard0from1[N/2]
    y = -L/2.0 + (N/4)*dx;
    u = 1.414 * y;
    c1_new[0][0]=.25*(guard1from13[0] + c1[1][0] + guard1from0[0] + c1[0][1])
    - dt/(2*dx) * (u* (c1[1][0]-guard1from13[0]) + v* (c1[0][1]-guard1from0[0]));


    //compute c1_new[0][N/4-1], need to use c[0][N/2]=guard0from1[0], c[-1][N/2-1]=guard0from2[N-1]
    y = -L/2.0 + dx * (N/2-1);
    u = 1.414 * y;
    c1_new[0][N/4-1]=.25*(guard1from13[N/4-1] + c1[1][N/4-1] + c1[0][N/4-2] + guard1from2[0])
    - dt/(2*dx) * ((u*(c1[1][N/4-1] - guard1from13[N/4-1])) + v* (guard1from2[0] - c1[0][N/4-2]));
    #pragma omp parallel for default(shared) schedule(static) num_threads(nt)
    //compute c1_new from row 1 to row N/4-2
    for(int i=1;i<N/4-1;i++)
       { 
        x= -L/2.0 + i*dx;
        v= -1.414 * x;

        //compute c_new[i][0]
        y= -L/2 + (N/4)*dx; 
        u= 1.414*y;
        c1_new[i][0]=.25*(c1[i-1][0] + c1[i+1][0] + guard1from0[i] + c1[i][1])
        - dt/(2*dx) * (u * (c1[i+1][0]-c1[i-1][0]) + v * (c1[i][1]-guard1from0[i]));
        //compute c0_new[i][j] for j=1,2..,N/4-2.
        for(int j=1;j<N/4-1;j++)
            {
            y = -L/2.0 + (N/4+j)*dx;
            u = 1.414 * y;
            c1_new[i][j]=.25*(c1[i-1][j] + c1[i+1][j] + c1[i][j-1] + c1[i][j+1]) 
            - dt/(2*dx) * (u * (c1[i+1][j]-c1[i-1][j]) + v * (c1[i][j+1]-c1[i][j-1]));
            }

        //compute c1_new[i][N/4-1]
        y=-L/2 + (N/2-1)*dx;
        u = 1.414*y;
        c1_new[i][N/4-1]=.25*(c1[i-1][N/4-1] + c1[i+1][N/4-1] + c1[i][N/4-2] + guard1from2[i]) 
        - dt/(2*dx) * (u * (c1[i+1][N/4-1]-c1[i-1][N/4-1]) + v * (guard1from2[i]-c1[i][N/4-2]));

        }

        //compute c1_new[N/4-1][*]. Need to use c[N/2][j] = guard0from2[j] for j = 1,2..N/2-2
        x = -L/2 + (N/4 - 1)*dx;
        v = -1.414 * x;
        for(int j=1; j<N/4-1; j++)
            {
            y = -L/2.0 + (N/4 + j)*dx;
            u = 1.414 * y;
            c1_new[N/4-1][j]=.25*(c1[N/4 - 2][j] + guard1from5[j] + c1[N/4 - 1][j-1] + c1[N/4 - 1][j+1])
            -dt/(2*dx) * (u * (guard1from5[j]-c1[N/4 - 2][j]) + v * (c1[N/4 - 1][j+1]-c1[N/4 - 1][j-1]));
            }
        //compute c1_new[N/4-1][0], c0[N/2][0]=guard0from2[0], c0[N/2-1][-1]=guard0from1[N-1]
        y = -L/2.0 + (N/4)*dx;
        u = 1.414 * y;
        c1_new[N/4-1][0]=.25*(c1[N/4 - 2][0] + guard1from5[0] + guard1from0[N/4-1] + c1[N/4 - 1][1])
        -dt/(2*dx) * (u * (guard1from5[0]-c1[N/4 - 2][0]) + v * (c1[N/4 - 1][1]-guard1from0[N/4-1]));

        //c1[N/4-1][N/4] = guard0from1[N/2-1], c0[N/2][N/2-1]=guard0from2[N/2-1]
        y = -L/2.0 + (N/2-1)*dx;
        u = 1.414 * y;
        c1_new[N/4-1][N/4-1]=.25*(c1[N/4 - 2][N/4-1] + guard1from5[N/4-1] + c1[N/4-1][N/4-2] + guard1from2[N/4-1])
        -dt/(2*dx) * (u * (guard1from5[N/4-1]-c1[N/4 - 2][N/4-1]) + v * (guard1from2[N/4-1]-c1[N/4-1][N/4-2]));

    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/4;i++)
            {
            for(int j=0;j<N/4;j++)
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

//write initial data to file
for(int i = 0; i<N/4; i++)
    { 
    for(int j=0; j<N/4; j++)  
        fprintf(data2,"%.4f ",c2[i][j]);
    fprintf(data2,"\n");
    }

t5 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #1 and #2
    for(int j=0;j<N/4;j++)
        {
        send2to1[j]=c2[j][0];
        send2to3[j]=c2[j][N/4-1];
        send2to6[j]=c2[N/4-1][j];
        send2to14[j]=c2[0][j];
        }

    //MPI COMM with #1
    MPI_Send(send2to1,N/4,MPI_DOUBLE,1,n,MPI_COMM_WORLD);
    MPI_Recv(guard2from1, N/4,MPI_DOUBLE,1,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #3
    MPI_Recv(guard2from3, N/4,MPI_DOUBLE,3,n,MPI_COMM_WORLD,&status);
    MPI_Send(send2to3,N/4,MPI_DOUBLE,3,n,MPI_COMM_WORLD);

    //MPI COMM with #6
    MPI_Recv(guard2from6, N/4,MPI_DOUBLE,6,n,MPI_COMM_WORLD,&status);
    MPI_Send(send2to6,N/4,MPI_DOUBLE,6,n,MPI_COMM_WORLD);

    //MPI COMM with #14
    MPI_Recv(guard2from14, N/4,MPI_DOUBLE,14,n,MPI_COMM_WORLD,&status);
    MPI_Send(send2to14,N/4,MPI_DOUBLE,14,n,MPI_COMM_WORLD);


    //compute c2_new[0][j].for j=1,2,..,N/2-2, c2[N/2-1][*]
    //c0[-1][j] = guard0from2[N/2 + j]
    x= -L/2;
    v=(-1.414)*x;
    for(int j=1;j<N/4-1;j++)
        {
        y = -L/2.0 + (N/2+j)*dx;
        u = 1.414 * y;
        c2_new[0][j]=.25*(guard2from14[j] + c2[1][j] + c2[0][j-1] + c2[0][j+1]) 
        - dt/(2*dx) * (u * (c2[1][j]-guard2from14[j]) + v * (c2[0][j+1]-c2[0][j-1]));
        }

    //compute c2_new[0][0] , need to use c[-1][0] = guard0from2[N/2] and c[0][-1]=guard0from1[N/2]
    y = -L/2.0 + (N/2)*dx;
    u = 1.414 * y;
    c2_new[0][0]=.25*(guard2from14[0] + c2[1][0] + guard2from1[0] + c2[0][1])
    - dt/(2*dx) * (u* (c2[1][0]-guard2from14[0]) + v* (c2[0][1]-guard2from1[0]));


    //compute c2_new[0][N/4-1], need to use c[0][N/2]=guard0from1[0], c[-1][N/2-1]=guard0from2[N-1]
    y = -L/2.0 + dx * (3*N/4-1);
    u = 1.414 * y;
    c2_new[0][N/4-1]=.25*(guard2from14[N/4-1] + c2[1][N/4-1] + c2[0][N/4-2] + guard2from3[0])
    - dt/(2*dx) * ((u*(c2[1][N/4-1] - guard2from14[N/4-1])) + v* (guard2from3[0] - c2[0][N/4-2]));
    #pragma omp parallel for default(shared) schedule(static) num_threads(nt)
    //compute c2_new from row 1 to row N/4-2
    for(int i=1;i<N/4-1;i++)
       { 
        x= -L/2.0 + i*dx;
        v= -1.414 * x;

        //compute c_new[i][0]
        y= -L/2 + (N/2)*dx; 
        u= 1.414*y;
        c2_new[i][0]=.25*(c2[i-1][0] + c2[i+1][0] + guard2from1[i] + c2[i][1])
        - dt/(2*dx) * (u * (c2[i+1][0]-c2[i-1][0]) + v * (c2[i][1]-guard2from1[i]));
        //compute c2_new[i][j] for j=1,2..,N/4-2.
        for(int j=1;j<N/4-1;j++)
            {
            y = -L/2.0 + (N/2+j)*dx;
            u = 1.414 * y;
            c2_new[i][j]=.25*(c2[i-1][j] + c2[i+1][j] + c2[i][j-1] + c2[i][j+1]) 
            - dt/(2*dx) * (u * (c2[i+1][j]-c2[i-1][j]) + v * (c2[i][j+1]-c2[i][j-1]));
            }

        //compute c2_new[i][N/4-1]
        y=-L/2 + (3*N/4-1)*dx;
        u = 1.414*y;
        c2_new[i][N/4-1]=.25*(c2[i-1][N/4-1] + c2[i+1][N/4-1] + c2[i][N/4-2] + guard2from3[i]) 
        - dt/(2*dx) * (u * (c2[i+1][N/4-1]-c2[i-1][N/4-1]) + v * (guard2from3[i]-c2[i][N/4-2]));

        }

        //compute c2_new[N/4-1][*]. Need to use c[N/2][j] = guard0from2[j] for j = 1,2..N/2-2
        x = -L/2 + (N/4 - 1)*dx;
        v = -1.414 * x;
        for(int j=1; j<N/4-1; j++)
            {
            y = -L/2.0 + (N/2 + j)*dx;
            u = 1.414 * y;
            c2_new[N/4-1][j]=.25*(c2[N/4 - 2][j] + guard2from6[j] + c2[N/4 - 1][j-1] + c2[N/4 - 1][j+1])
            -dt/(2*dx) * (u * (guard2from6[j]-c2[N/4 - 2][j]) + v * (c2[N/4 - 1][j+1]-c2[N/4 - 1][j-1]));
            }
        //compute c1_new[N/4-1][0], c0[N/2][0]=guard0from2[0], c0[N/2-1][-1]=guard0from1[N-1]
        y = -L/2.0 + (N/2)*dx;
        u = 1.414 * y;
        c2_new[N/4-1][0]=.25*(c2[N/4 - 2][0] + guard2from6[0] + guard2from1[N/4-1] + c2[N/4 - 1][1])
        -dt/(2*dx) * (u * (guard2from6[0]-c2[N/4 - 2][0]) + v * (c2[N/4 - 1][1]-guard2from1[N/4-1]));

        //c1[N/4-1][N/4] = guard0from1[N/2-1], c0[N/2][N/2-1]=guard0from2[N/2-1]
        y = -L/2.0 + (3*N/4-1)*dx;
        u = 1.414 * y;
        c2_new[N/4-1][N/4-1]=.25*(c2[N/4 - 2][N/4-1] + guard2from6[N/4-1] + c2[N/4-1][N/4-2] + guard2from3[N/4-1])
        -dt/(2*dx) * (u * (guard2from6[N/4-1]-c2[N/4 - 2][N/4-1]) + v * (guard2from3[N/4-1]-c2[N/4-1][N/4-2]));

    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/4;i++)
            {
            for(int j=0;j<N/4;j++)
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

//write initial data to file
for(int i = 0; i<N/4; i++)
    { 
    for(int j=0; j<N/4; j++)  
        fprintf(data3,"%.4f ",c3[i][j]);
    fprintf(data3,"\n");
    }

t7 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #1 and #2
    for(int j=0;j<N/4;j++)
        {
        send3to0[j]=c3[j][N/4-1];
        send3to2[j]=c3[j][0];
        send3to7[j]=c3[N/4-1][j];
        send3to15[j]=c3[0][j];
        }

    //MPI COMM with #0
    MPI_Send(send3to0,N/4,MPI_DOUBLE,0,n,MPI_COMM_WORLD);
    MPI_Recv(guard3from0, N/4,MPI_DOUBLE,0,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #2
    MPI_Send(send3to2,N/4,MPI_DOUBLE,2,n,MPI_COMM_WORLD);
    MPI_Recv(guard3from2, N/4,MPI_DOUBLE,2,n,MPI_COMM_WORLD,&status);
 

    //MPI COMM with #7
    MPI_Recv(guard3from7, N/4,MPI_DOUBLE,7,n,MPI_COMM_WORLD,&status);
    MPI_Send(send3to7,N/4,MPI_DOUBLE,7,n,MPI_COMM_WORLD);

    //MPI COMM with #15
    MPI_Recv(guard3from15, N/4,MPI_DOUBLE,15,n,MPI_COMM_WORLD,&status);
    MPI_Send(send3to15,N/4,MPI_DOUBLE,15,n,MPI_COMM_WORLD);


    //compute c3_new[0][j].for j=1,2,..,N/2-2, c2[N/2-1][*]
    //c0[-1][j] = guard0from2[N/2 + j]
    x= -L/2;
    v=(-1.414)*x;
    for(int j=1;j<N/4-1;j++)
        {
        y = -L/2.0 + (3*N/4 + j)*dx;
        u = 1.414 * y;
        c3_new[0][j]=.25*(guard3from15[j] + c3[1][j] + c3[0][j-1] + c3[0][j+1]) 
        - dt/(2*dx) * (u * (c3[1][j]-guard3from15[j]) + v * (c3[0][j+1]-c3[0][j-1]));
        }

    //compute c3_new[0][0] , need to use c[-1][0] = guard0from2[N/2] and c[0][-1]=guard0from1[N/2]
    y = -L/2.0 + (3*N/4)*dx;
    u = 1.414 * y;
    c3_new[0][0]=.25*(guard3from15[0] + c3[1][0] + guard3from2[0] + c2[0][1])
    - dt/(2*dx) * (u* (c3[1][0]-guard3from15[0]) + v* (c2[0][1]-guard3from2[0]));


    //compute c2_new[0][N/4-1], need to use c[0][N/2]=guard0from1[0], c[-1][N/2-1]=guard0from2[N-1]
    y = -L/2.0 + dx * (N-1);
    u = 1.414 * y;
    c3_new[0][N/4-1]=.25*(guard3from15[N/4-1] + c3[1][N/4-1] + c3[0][N/4-2] + guard3from0[0])
    - dt/(2*dx) * ((u*(c3[1][N/4-1] - guard3from15[N/4-1])) + v* (guard3from0[0] - c3[0][N/4-2]));

    //compute c3_new from row 1 to row N/4-2
    #pragma omp parallel for default(shared) schedule(static) num_threads(nt)
    for(int i=1;i<N/4-1;i++)
       { 
        x= -L/2.0 + i*dx;
        v= -1.414 * x;

        //compute c3_new[i][0]
        y= -L/2 + (3*N/4)*dx; 
        u= 1.414*y;
        c3_new[i][0]=.25*(c3[i-1][0] + c3[i+1][0] + guard3from2[i] + c3[i][1])
        - dt/(2*dx) * (u * (c3[i+1][0]-c3[i-1][0]) + v * (c3[i][1]-guard3from2[i]));
        //compute c3_new[i][j] for j=1,2..,N/4-2.
        for(int j=1;j<N/4-1;j++)
            {
            y = -L/2.0 + (3*N/4+j)*dx;
            u = 1.414 * y;
            c3_new[i][j]=.25*(c3[i-1][j] + c3[i+1][j] + c3[i][j-1] + c3[i][j+1]) 
            - dt/(2*dx) * (u * (c3[i+1][j]-c3[i-1][j]) + v * (c3[i][j+1]-c3[i][j-1]));
            }

        //compute c3_new[i][N/4-1]
        y=-L/2 + (N-1)*dx;
        u = 1.414*y;
        c3_new[i][N/4-1]=.25*(c3[i-1][N/4-1] + c3[i+1][N/4-1] + c3[i][N/4-2] + guard3from0[i]) 
        - dt/(2*dx) * (u * (c3[i+1][N/4-1]-c3[i-1][N/4-1]) + v * (guard3from0[i]-c3[i][N/4-2]));

        }

        //compute c3_new[N/4-1][*]. Need to use c[N/2][j] = guard0from2[j] for j = 1,2..N/2-2
        x = -L/2 + (N/4 - 1)*dx;
        v = -1.414 * x;
        for(int j=1; j<N/4-1; j++)
            {
            y = -L/2.0 + (3*N/4 + j)*dx;
            u = 1.414 * y;
            c3_new[N/4-1][j]=.25*(c3[N/4 - 2][j] + guard3from7[j] + c3[N/4 - 1][j-1] + c3[N/4 - 1][j+1])
            -dt/(2*dx) * (u * (guard3from7[j]-c3[N/4 - 2][j]) + v * (c3[N/4 - 1][j+1]-c3[N/4 - 1][j-1]));
            }
        //compute c3_new[N/4-1][0], c0[N/2][0]=guard0from2[0], c0[N/2-1][-1]=guard0from1[N-1]
        y = -L/2.0 + (3*N/4)*dx;
        u = 1.414 * y;
        c3_new[N/4-1][0]=.25*(c3[N/4 - 2][0] + guard3from7[0] + guard3from2[N/4-1] + c3[N/4 - 1][1])
        -dt/(2*dx) * (u * (guard3from7[0]-c3[N/4 - 2][0]) + v * (c3[N/4 - 1][1]-guard3from2[N/4-1]));

        //c1[N/4-1][N/4] = guard0from1[N/2-1], c0[N/2][N/2-1]=guard0from2[N/2-1]
        y = -L/2.0 + (N-1)*dx;
        u = 1.414 * y;
        c3_new[N/4-1][N/4-1]=.25*(c3[N/4 - 2][N/4-1] + guard3from7[N/4-1] + c3[N/4-1][N/4-2] + guard3from0[N/4-1])
        -dt/(2*dx) * (u * (guard3from7[N/4-1]-c3[N/4 - 2][N/4-1]) + v * (guard3from0[N/4-1]-c3[N/4-1][N/4-2]));

    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/4;i++)
            {
            for(int j=0;j<N/4;j++)
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

if(mype==4)
{
FILE *data4=fopen("lax4.txt", "w");
if(data4==NULL)
    {
    printf("Error opening file!\n");
    return 1;  // Indicate failure
    }


//initiate the 1s in c1[][]
for(int i = 3*N/20; i<N/4; i++)
    for(int j=0; j<N/4; j++)
        c4[i][j] = 1;


//write initial data to file
for(int i = 0; i<N/4; i++)
    { 
    for(int j=0; j<N/4; j++)  
        fprintf(data4,"%.4f ",c4[i][j]);
    fprintf(data4,"\n");
    }

t9 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #1 and #2
    for(int j=0;j<N/4;j++)
        {
        send4to0[j]=c4[0][j];
        send4to5[j]=c4[j][N/4-1];
        send4to7[j]=c4[j][0];
        send4to8[j]=c4[N/4-1][j];
        }

    //MPI COMM with #0
    MPI_Send(send4to0,N/4,MPI_DOUBLE,0,n,MPI_COMM_WORLD);
    MPI_Recv(guard4from0, N/4,MPI_DOUBLE,0,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #5
    MPI_Recv(guard4from5, N/4,MPI_DOUBLE,5,n,MPI_COMM_WORLD,&status);
    MPI_Send(send4to5,N/4,MPI_DOUBLE,5,n,MPI_COMM_WORLD);

    //MPI COMM with #7
    MPI_Recv(guard4from7, N/4,MPI_DOUBLE,7,n,MPI_COMM_WORLD,&status);
    MPI_Send(send4to7,N/4,MPI_DOUBLE,7,n,MPI_COMM_WORLD);

    //MPI COMM with #8
    MPI_Recv(guard4from8, N/4,MPI_DOUBLE,8,n,MPI_COMM_WORLD,&status);
    MPI_Send(send4to8,N/4,MPI_DOUBLE,8,n,MPI_COMM_WORLD);


    //compute c4_new[0][j].for j=1,2,..,N/2-2, c2[N/2-1][*]
    //c4[-1][j] = guard0from2[N/2 + j]
    x= -L/2.0 + (N/4+0)*dx;
    v=(-1.414)*x;
    for(int j=1;j<N/4-1;j++)
    {
    y = -L/2.0 + j*dx;
    u = 1.414 * y;
    c4_new[0][j]=.25*(guard4from0[j] + c4[1][j] + c4[0][j-1] + c4[0][j+1]) 
    - dt/(2*dx) * (u * (c4[1][j]-guard4from0[j]) + v * (c4[0][j+1]-c4[0][j-1]));
    }

    //compute c4_new[0][0] , need to use c[-1][0] = guard0from2[N/2] and c[0][-1]=guard0from1[N/2]
    y = -L/2.0;
    u = 1.414 * y;
    c4_new[0][0]=.25*(guard4from0[0] + c4[1][0] + guard4from7[0] + c4[0][1])
    - dt/(2*dx) * (u* (c4[1][0]-guard4from0[0]) + v* (c4[0][1]-guard4from7[0]));

    //compute c4_new[0][N/4-1], need to use c[0][N/2]=guard0from1[0], c[-1][N/2-1]=guard0from2[N-1]
    y = -L/2.0 + dx * (N/4-1);
    u = 1.414 * y;
    c4_new[0][N/4-1]=.25*(guard4from0[N/4-1] + c4[1][N/4-1] + c4[0][N/4-2] + guard4from5[0])
    - dt/(2*dx) * ((u*(c4[1][N/4-1] - guard4from0[N/4-1])) + v* (guard4from5[0] - c4[0][N/4-2]));

    #pragma omp parallel for default(shared) schedule(static) num_threads(nt)
    //compute c4_new from row 1 to row N/4-2
    for(int i=1;i<N/4-1;i++)
       { 
        x= -L/2.0 + (N/4+i)*dx;
        v= -1.414 * x;

        //compute c_new[i][0]
        y= -L/2; 
        u= 1.414*y;
        c4_new[i][0]=.25*(c4[i-1][0] + c4[i+1][0] + guard4from7[i] + c4[i][1])
        - dt/(2*dx) * (u * (c4[i+1][0]-c4[i-1][0]) + v * (c4[i][1]-guard4from7[i]));
        //compute c0_new[i][j] for j=1,2..,N/4-2.
        for(int j=1;j<N/4-1;j++)
            {
            y = -L/2.0 + j*dx;
            u = 1.414 * y;
            c4_new[i][j]=.25*(c4[i-1][j] + c4[i+1][j] + c4[i][j-1] + c4[i][j+1]) 
            - dt/(2*dx) * (u * (c4[i+1][j]-c4[i-1][j]) + v * (c4[i][j+1]-c4[i][j-1]));
            }

        //compute c4_new[i][N/4-1]
        y=-L/2 + (N/4-1)*dx;
        u = 1.414*y;
        c4_new[i][N/4-1]=.25*(c4[i-1][N/4-1] + c4[i+1][N/4-1] + c4[i][N/4-2] + guard4from5[i]) 
        - dt/(2*dx) * (u * (c4[i+1][N/4-1]-c4[i-1][N/4-1]) + v * (guard4from5[i]-c4[i][N/4-2]));

        }

        //compute c4_new[N/4-1][j]for j = 1,2..N/2-2
        x = -L/2 + (N/2 - 1)*dx;
        v = -1.414 * x;
        for(int j=1; j<N/4-1; j++)
            {
            y = -L/2.0 + j*dx;
            u = 1.414 * y;
            c4_new[N/4-1][j]=.25*(c4[N/4 - 2][j] + guard4from8[j] + c4[N/4 - 1][j-1] + c4[N/4 - 1][j+1])
            -dt/(2*dx) * (u * (guard4from8[j]-c4[N/4 - 2][j]) + v * (c4[N/4 - 1][j+1]-c4[N/4 - 1][j-1]));
            }
        //compute c0_new[N/4-1][0], c0[N/2][0]=guard0from2[0], c0[N/2-1][-1]=guard0from1[N-1]
        y = -L/2.0;
        u = 1.414 * y;
        c4_new[N/4-1][0]=.25*(c4[N/4 - 2][0] + guard4from8[0] + guard4from7[N/4-1] + c4[N/4 - 1][1])
        -dt/(2*dx) * (u * (guard4from8[0]-c4[N/4 - 2][0]) + v * (c4[N/4 - 1][1]-guard4from7[N/4-1]));

        //c0[N/2-1][N/2] = guard0from1[N/2-1], c0[N/2][N/2-1]=guard0from2[N/2-1]
        y = -L/2.0 + (N/4-1)*dx;
        u = 1.414 * y;
        c4_new[N/4-1][N/4-1]=.25*(c4[N/4 - 2][N/4-1] + guard4from8[N/4-1] + c4[N/4-1][N/4-2] + guard4from5[N/4-1])
        -dt/(2*dx) * (u * (guard4from8[N/4-1]-c4[N/4 - 2][N/4-1]) + v * (guard4from5[N/4-1]-c4[N/4-1][N/4-2]));

    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/2;i++)
            {
            for(int j=0;j<N/2;j++)
                fprintf(data4,"%.4f ",c4_new[i][j]); 
            fprintf(data4,"\n"); 
            }
       } 

    pt4=c4;
    c4=c4_new;
    c4_new=pt4;
    }
t10 = omp_get_wtime();
fclose(data4);
}


else if(mype==5)
{
FILE *data5=fopen("lax5.txt", "w");
if(data5==NULL)
    {
    printf("Error opening file!\n");
    return 1;  // Indicate failure
    }

//initiate the 1s in c1[][]
for(int i = 3*N/20; i<N/4; i++)
    for(int j=0; j<N/4; j++)
        c5[i][j] = 1;

//write initial data to file
for(int i = 0; i<N/4; i++)
    { 
    for(int j=0; j<N/4; j++)  
        fprintf(data5,"%.4f ",c5[i][j]);
    fprintf(data5,"\n");
    }

t11 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #1 and #2
    for(int j=0;j<N/4;j++)
        {
        send5to1[j]=c5[0][j];
        send5to4[j]=c5[j][0];
        send5to6[j]=c5[j][N/4-1];
        send5to9[j]=c5[N/4-1][j];
        }

    //MPI COMM with #1
    MPI_Send(send5to1,N/4,MPI_DOUBLE,1,n,MPI_COMM_WORLD);
    MPI_Recv(guard1from5, N/4,MPI_DOUBLE,1,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #4
    MPI_Send(send5to4,N/4,MPI_DOUBLE,4,n,MPI_COMM_WORLD);
    MPI_Recv(guard5from4, N/4,MPI_DOUBLE,4,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #6
    MPI_Recv(guard5from6, N/4,MPI_DOUBLE,6,n,MPI_COMM_WORLD,&status);
    MPI_Send(send5to6,N/4,MPI_DOUBLE,6,n,MPI_COMM_WORLD);

    //MPI COMM with #9
    MPI_Recv(guard5from9, N/4,MPI_DOUBLE,9,n,MPI_COMM_WORLD,&status);
    MPI_Send(send5to9,N/4,MPI_DOUBLE,9,n,MPI_COMM_WORLD);


    //compute c5_new[0][j].for j=1,2,..,N/2-2, c2[N/2-1][*]
    //c5[-1][j] = guard0from2[N/2 + j]
    x= -L/2 + dx *(N/4 + 0);
    v=(-1.414)*x;
    for(int j=1;j<N/4-1;j++)
        {
        y = -L/2.0 + (N/4+j)*dx;
        u = 1.414 * y;
        c5_new[0][j]=.25*(guard5from1[j] + c5[1][j] + c5[0][j-1] + c5[0][j+1]) 
        - dt/(2*dx) * (u * (c5[1][j]-guard5from1[j]) + v * (c5[0][j+1]-c5[0][j-1]));
        }

    //compute c1_new[0][0] , need to use c[-1][0] = guard0from2[N/2] and c[0][-1]=guard0from1[N/2]
    y = -L/2.0 + (N/4)*dx;
    u = 1.414 * y;
    c5_new[0][0]=.25*(guard5from1[0] + c5[1][0] + guard5from4[0] + c5[0][1])
    - dt/(2*dx) * (u* (c5[1][0]-guard5from1[0]) + v* (c5[0][1]-guard5from4[0]));


    //compute c5_new[0][N/4-1], need to use c[0][N/2]=guard0from1[0], c[-1][N/2-1]=guard0from2[N-1]
    y = -L/2.0 + dx * (N/2-1);
    u = 1.414 * y;
    c5_new[0][N/4-1]=.25*(guard5from1[N/4-1] + c5[1][N/4-1] + c5[0][N/4-2] + guard5from6[0])
    - dt/(2*dx) * ((u*(c5[1][N/4-1] - guard5from1[N/4-1])) + v* (guard5from6[0] - c5[0][N/4-2]));

    #pragma omp parallel for default(shared) schedule(static) num_threads(nt)
    //compute c5_new from row 1 to row N/4-2
    for(int i=1;i<N/4-1;i++)
       { 
        x= -L/2.0 + (N/4+i)*dx;
        v= -1.414 * x;

        //compute c5_new[i][0]
        y= -L/2 + (N/4)*dx; 
        u= 1.414*y;
        c5_new[i][0]=.25*(c5[i-1][0] + c5[i+1][0] + guard5from4[i] + c5[i][1])
        - dt/(2*dx) * (u * (c5[i+1][0]-c5[i-1][0]) + v * (c5[i][1]-guard5from4[i]));
        //compute c0_new[i][j] for j=1,2..,N/4-2.
        for(int j=1;j<N/4-1;j++)
            {
            y = -L/2.0 + (N/4+j)*dx;
            u = 1.414 * y;
            c5_new[i][j]=.25*(c5[i-1][j] + c5[i+1][j] + c5[i][j-1] + c5[i][j+1]) 
            - dt/(2*dx) * (u * (c5[i+1][j]-c5[i-1][j]) + v * (c5[i][j+1]-c5[i][j-1]));
            }

        //compute c1_new[i][N/4-1]
        y=-L/2 + (N/2-1)*dx;
        u = 1.414*y;
        c5_new[i][N/4-1]=.25*(c5[i-1][N/4-1] + c5[i+1][N/4-1] + c5[i][N/4-2] + guard5from6[i]) 
        - dt/(2*dx) * (u * (c5[i+1][N/4-1]-c5[i-1][N/4-1]) + v * (guard5from6[i]-c5[i][N/4-2]));

        }

        //compute c5_new[N/4-1][*]. Need to use c[N/2][j] = guard0from2[j] for j = 1,2..N/2-2
        x = -L/2 + (N/2 - 1)*dx;
        v = -1.414 * x;
        for(int j=1; j<N/4-1; j++)
            {
            y = -L/2.0 + (N/4 + j)*dx;
            u = 1.414 * y;
            c5_new[N/4-1][j]=.25*(c5[N/4 - 2][j] + guard5from9[j] + c5[N/4 - 1][j-1] + c5[N/4 - 1][j+1])
            -dt/(2*dx) * (u * (guard5from9[j]-c5[N/4 - 2][j]) + v * (c5[N/4 - 1][j+1]-c5[N/4 - 1][j-1]));
            }
        //compute c5_new[N/4-1][0], c0[N/2][0]=guard0from2[0], c0[N/2-1][-1]=guard0from1[N-1]
        y = -L/2.0 + (N/4)*dx;
        u = 1.414 * y;
        c5_new[N/4-1][0]=.25*(c5[N/4 - 2][0] + guard5from9[0] + guard5from4[N/4-1] + c5[N/4 - 1][1])
        -dt/(2*dx) * (u * (guard5from9[0]-c5[N/4 - 2][0]) + v * (c5[N/4 - 1][1]-guard5from4[N/4-1]));

        //c1[N/4-1][N/4] = guard0from1[N/2-1], c0[N/2][N/2-1]=guard0from2[N/2-1]
        y = -L/2.0 + (N/2-1)*dx;
        u = 1.414 * y;
        c5_new[N/4-1][N/4-1]=.25*(c5[N/4 - 2][N/4-1] + guard5from9[N/4-1] + c5[N/4-1][N/4-2] + guard5from6[N/4-1])
        -dt/(2*dx) * (u * (guard5from9[N/4-1]-c5[N/4 - 2][N/4-1]) + v * (guard5from6[N/4-1]-c5[N/4-1][N/4-2]));

    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/4;i++)
            {
            for(int j=0;j<N/4;j++)
                fprintf(data5,"%.4f ",c5_new[i][j]); 
            fprintf(data5,"\n"); 
            }
       } 

    pt5=c5;
    c5=c5_new;
    c5_new=pt5;
    }
t12 = omp_get_wtime();
fclose(data5);
}


else if(mype==6)
{
FILE *data6=fopen("lax6.txt", "w");
if(data6==NULL)
    {
    printf("Error opening file!\n");
    return 1;  // Indicate failure
    }


for(int i = 3*N/20; i<N/4; i++)
    for(int j=0; j<N/4; j++)
        c6[i][j] = 1;
//write initial data to file
for(int i = 0; i<N/4; i++)
    { 
    for(int j=0; j<N/4; j++)  
        fprintf(data6,"%.4f ",c2[i][j]);
    fprintf(data6,"\n");
    }

t13 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #1 and #2
    for(int j=0;j<N/4;j++)
        {
        send6to2[j]=c6[0][j];
        send6to5[j]=c6[j][0];
        send6to7[j]=c6[j][N/4-1];
        send6to10[j]=c6[N/4-1][j];
        }

    //MPI COMM with #2
    MPI_Send(send6to2,N/4,MPI_DOUBLE,2,n,MPI_COMM_WORLD);
    MPI_Recv(guard6from2, N/4,MPI_DOUBLE,2,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #5
    MPI_Send(send6to5,N/4,MPI_DOUBLE,5,n,MPI_COMM_WORLD);
    MPI_Recv(guard6from5, N/4,MPI_DOUBLE,5,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #7
    MPI_Recv(guard6from7, N/4,MPI_DOUBLE,7,n,MPI_COMM_WORLD,&status);
    MPI_Send(send6to7,N/4,MPI_DOUBLE,7,n,MPI_COMM_WORLD);

    //MPI COMM with #10
    MPI_Recv(guard6from10, N/4,MPI_DOUBLE,10,n,MPI_COMM_WORLD,&status);
    MPI_Send(send6to10,N/4,MPI_DOUBLE,10,n,MPI_COMM_WORLD);


    //compute c6_new[0][j].for j=1,2,..,N/2-2, c2[N/2-1][*]
    //c0[-1][j] = guard0from2[N/2 + j]
    x= -L/2 + (N/4)*dx;
    v=(-1.414)*x;
    for(int j=1;j<N/4-1;j++)
        {
        y = -L/2.0 + (N/2+j)*dx;
        u = 1.414 * y;
        c6_new[0][j]=.25*(guard6from2[j] + c6[1][j] + c6[0][j-1] + c6[0][j+1]) 
        - dt/(2*dx) * (u * (c6[1][j]-guard6from2[j]) + v * (c6[0][j+1]-c6[0][j-1]));
        }

    //compute c2_new[0][0] , need to use c[-1][0] = guard0from2[N/2] and c[0][-1]=guard0from1[N/2]
    y = -L/2.0 + (N/2)*dx;
    u = 1.414 * y;
    c6_new[0][0]=.25*(guard6from2[0] + c6[1][0] + guard6from5[0] + c6[0][1])
    - dt/(2*dx) * (u* (c6[1][0]-guard6from2[0]) + v* (c6[0][1]-guard6from5[0]));


    //compute c2_new[0][N/4-1], need to use c[0][N/2]=guard0from1[0], c[-1][N/2-1]=guard0from2[N-1]
    y = -L/2.0 + dx * (3*N/4-1);
    u = 1.414 * y;
    c6_new[0][N/4-1]=.25*(guard6from2[N/4-1] + c6[1][N/4-1] + c6[0][N/4-2] + guard6from7[0])
    - dt/(2*dx) * ((u*(c6[1][N/4-1] - guard6from2[N/4-1])) + v* (guard6from7[0] - c6[0][N/4-2]));

    #pragma omp parallel for default(shared) schedule(static) num_threads(nt)
    //compute c6_new from row 1 to row N/4-2
    for(int i=1;i<N/4-1;i++)
       { 
        x= -L/2.0 + (N/4+i)*dx;
        v= -1.414 * x;
        //compute c_new[i][0]
        y= -L/2 + (N/2)*dx; 
        u= 1.414*y;
        c6_new[i][0]=.25*(c6[i-1][0] + c6[i+1][0] + guard6from5[i] + c6[i][1])
        - dt/(2*dx) * (u * (c6[i+1][0]-c6[i-1][0]) + v * (c6[i][1]-guard6from5[i]));
        //compute c6_new[i][j] for j=1,2..,N/4-2.
        for(int j=1;j<N/4-1;j++)
            {
            y = -L/2.0 + (N/2+j)*dx;
            u = 1.414 * y;
            c6_new[i][j]=.25*(c6[i-1][j] + c6[i+1][j] + c6[i][j-1] + c6[i][j+1]) 
            - dt/(2*dx) * (u * (c6[i+1][j]-c6[i-1][j]) + v * (c6[i][j+1]-c6[i][j-1]));
            }

        //compute c6_new[i][N/4-1]
        y=-L/2 + (3*N/4-1)*dx;
        u = 1.414*y;
        c6_new[i][N/4-1]=.25*(c6[i-1][N/4-1] + c6[i+1][N/4-1] + c6[i][N/4-2] + guard6from7[i]) 
        - dt/(2*dx) * (u * (c6[i+1][N/4-1]-c6[i-1][N/4-1]) + v * (guard6from7[i]-c6[i][N/4-2]));

        }

        //compute c6_new[N/4-1][*]. Need to use c[N/2][j] = guard0from2[j] for j = 1,2..N/2-2
        x = -L/2 + (N/2 - 1)*dx;
        v = -1.414 * x;
        for(int j=1; j<N/4-1; j++)
            {
            y = -L/2.0 + (N/2 + j)*dx;
            u = 1.414 * y;
            c6_new[N/4-1][j]=.25*(c6[N/4 - 2][j] + guard6from10[j] + c6[N/4 - 1][j-1] + c6[N/4 - 1][j+1])
            -dt/(2*dx) * (u * (guard6from10[j]-c6[N/4 - 2][j]) + v * (c6[N/4 - 1][j+1]-c6[N/4 - 1][j-1]));
            }
        //compute c6_new[N/4-1][0], c0[N/2][0]=guard0from2[0], c0[N/2-1][-1]=guard0from1[N-1]
        y = -L/2.0 + (N/2)*dx;
        u = 1.414 * y;
        c6_new[N/4-1][0]=.25*(c6[N/4 - 2][0] + guard6from10[0] + guard6from5[N/4-1] + c6[N/4 - 1][1])
        -dt/(2*dx) * (u * (guard6from10[0]-c6[N/4 - 2][0]) + v * (c6[N/4 - 1][1]-guard6from5[N/4-1]));

        //c1[N/4-1][N/4] = guard0from1[N/2-1], c0[N/2][N/2-1]=guard0from2[N/2-1]
        y = -L/2.0 + (3*N/4-1)*dx;
        u = 1.414 * y;
        c6_new[N/4-1][N/4-1]=.25*(c6[N/4 - 2][N/4-1] + guard6from10[N/4-1] + c6[N/4-1][N/4-2] + guard6from7[N/4-1])
        -dt/(2*dx) * (u * (guard6from10[N/4-1]-c6[N/4 - 2][N/4-1]) + v * (guard6from7[N/4-1]-c6[N/4-1][N/4-2]));

    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/4;i++)
            {
            for(int j=0;j<N/4;j++)
                fprintf(data6,"%.4f ",c6_new[i][j]); 
            fprintf(data6,"\n"); 
            }
       } 

    pt6=c6;
    c6=c6_new;
    c6_new=pt6;
    }
t14 = omp_get_wtime();
fclose(data6);
}

else if(mype==7)
{
FILE *data7=fopen("lax7.txt", "w");
if(data7==NULL)
    {
    printf("Error opening file!\n");
    return 1;  // Indicate failure
    }

for(int i = 3*N/20; i<N/4; i++)
    for(int j=0; j<N/4; j++)
        c7[i][j] = 1;

//write initial data to file
for(int i = 0; i<N/4; i++)
    { 
    for(int j=0; j<N/4; j++)  
        fprintf(data7,"%.4f ",c7[i][j]);
    fprintf(data7,"\n");
    }

t15 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #1 and #2
    for(int j=0;j<N/4;j++)
        {
        send7to3[j]=c7[0][j];
        send7to4[j]=c7[j][N/4-1];
        send7to6[j]=c7[j][0];
        send7to11[j]=c7[N/4-1][j];
        }

    //MPI COMM with #3
    MPI_Send(send7to3,N/4,MPI_DOUBLE,3,n,MPI_COMM_WORLD);
    MPI_Recv(guard7from3, N/4,MPI_DOUBLE,3,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #4
    MPI_Send(send7to4,N/4,MPI_DOUBLE,4,n,MPI_COMM_WORLD);
    MPI_Recv(guard7from4, N/4,MPI_DOUBLE,4,n,MPI_COMM_WORLD,&status);
 

    //MPI COMM with #6
    MPI_Send(send7to6,N/4,MPI_DOUBLE,6,n,MPI_COMM_WORLD);
    MPI_Recv(guard7from6, N/4,MPI_DOUBLE,6,n,MPI_COMM_WORLD,&status);


    //MPI COMM with #11
    MPI_Recv(guard7from11, N/4,MPI_DOUBLE,11,n,MPI_COMM_WORLD,&status);
    MPI_Send(send7to11,N/4,MPI_DOUBLE,11,n,MPI_COMM_WORLD);


    //compute c7_new[0][j].for j=1,2,..,N/2-2, c2[N/2-1][*]
    //c0[-1][j] = guard0from2[N/2 + j]
    x= -L/2 + (N/4) * dx;
    v=(-1.414)*x;
    for(int j=1;j<N/4-1;j++)
        {
        y = -L/2.0 + (3*N/4 + j)*dx;
        u = 1.414 * y;
        c7_new[0][j]=.25*(guard7from3[j] + c7[1][j] + c7[0][j-1] + c7[0][j+1]) 
        - dt/(2*dx) * (u * (c7[1][j]-guard7from3[j]) + v * (c7[0][j+1]-c7[0][j-1]));
        }

    //compute c7_new[0][0] , need to use c[-1][0] = guard0from2[N/2] and c[0][-1]=guard0from1[N/2]
    y = -L/2.0 + (3*N/4)*dx;
    u = 1.414 * y;
    c7_new[0][0]=.25*(guard7from3[0] + c7[1][0] + guard7from6[0] + c7[0][1])
    - dt/(2*dx) * (u* (c7[1][0]-guard7from3[0]) + v* (c7[0][1]-guard7from6[0]));


    //compute c7_new[0][N/4-1], need to use c[0][N/2]=guard0from1[0], c[-1][N/2-1]=guard0from2[N-1]
    y = -L/2.0 + dx * (N-1);
    u = 1.414 * y;
    c7_new[0][N/4-1]=.25*(guard7from3[N/4-1] + c7[1][N/4-1] + c7[0][N/4-2] + guard7from4[0])
    - dt/(2*dx) * ((u*(c7[1][N/4-1] - guard7from3[N/4-1])) + v* (guard7from4[0] - c7[0][N/4-2]));

    #pragma omp parallel for default(shared) schedule(static) num_threads(nt)
    //compute c7_new from row 1 to row N/4-2
    for(int i=1;i<N/4-1;i++)
       { 
        x= -L/2.0 + (N/4+i)*dx;
        v= -1.414 * x;

        //compute c3_new[i][0]
        y= -L/2 + (3*N/4)*dx; 
        u= 1.414*y;
        c7_new[i][0]=.25*(c7[i-1][0] + c7[i+1][0] + guard7from6[i] + c7[i][1])
        - dt/(2*dx) * (u * (c7[i+1][0]-c7[i-1][0]) + v * (c7[i][1]-guard7from6[i]));
        //compute c3_new[i][j] for j=1,2..,N/4-2.
        for(int j=1;j<N/4-1;j++)
            {
            y = -L/2.0 + (3*N/4+j)*dx;
            u = 1.414 * y;
            c7_new[i][j]=.25*(c7[i-1][j] + c7[i+1][j] + c7[i][j-1] + c7[i][j+1]) 
            - dt/(2*dx) * (u * (c7[i+1][j]-c7[i-1][j]) + v * (c7[i][j+1]-c7[i][j-1]));
            }

        //compute c3_new[i][N/4-1]
        y=-L/2 + (N-1)*dx;
        u = 1.414*y;
        c7_new[i][N/4-1]=.25*(c7[i-1][N/4-1] + c7[i+1][N/4-1] + c7[i][N/4-2] + guard7from4[i]) 
        - dt/(2*dx) * (u * (c7[i+1][N/4-1]-c7[i-1][N/4-1]) + v * (guard7from4[i]-c7[i][N/4-2]));

        }

        //compute c7_new[N/4-1][*]. Need to use c[N/2][j] = guard0from2[j] for j = 1,2..N/2-2
        x = -L/2 + (N/2 - 1)*dx;
        v = -1.414 * x;
        for(int j=1; j<N/4-1; j++)
            {
            y = -L/2.0 + (3*N/4 + j)*dx;
            u = 1.414 * y;
            c7_new[N/4-1][j]=.25*(c7[N/4 - 2][j] + guard7from11[j] + c7[N/4 - 1][j-1] + c7[N/4 - 1][j+1])
            -dt/(2*dx) * (u * (guard7from11[j]-c7[N/4 - 2][j]) + v * (c7[N/4 - 1][j+1]-c7[N/4 - 1][j-1]));
            }
        //compute c7_new[N/4-1][0], c0[N/2][0]=guard0from2[0], c0[N/2-1][-1]=guard0from1[N-1]
        y = -L/2.0 + (3*N/4)*dx;
        u = 1.414 * y;
        c7_new[N/4-1][0]=.25*(c7[N/4 - 2][0] + guard7from11[0] + guard7from6[N/4-1] + c7[N/4 - 1][1])
        -dt/(2*dx) * (u * (guard7from11[0]-c7[N/4 - 2][0]) + v * (c7[N/4 - 1][1]-guard7from6[N/4-1]));

        //c1[N/4-1][N/4] = guard0from1[N/2-1], c0[N/2][N/2-1]=guard0from2[N/2-1]
        y = -L/2.0 + (N-1)*dx;
        u = 1.414 * y;
        c7_new[N/4-1][N/4-1]=.25*(c7[N/4 - 2][N/4-1] + guard7from11[N/4-1] + c7[N/4-1][N/4-2] + guard7from4[N/4-1])
        -dt/(2*dx) * (u * (guard7from11[N/4-1]-c7[N/4 - 2][N/4-1]) + v * (guard7from4[N/4-1]-c7[N/4-1][N/4-2]));

    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/4;i++)
            {
            for(int j=0;j<N/4;j++)
                fprintf(data7,"%.4f ",c7_new[i][j]); 
            fprintf(data7,"\n"); 
            }
       } 

    pt7=c7;
    c7=c7_new;
    c7_new=pt7;
    }
t16 = omp_get_wtime();
fclose(data7);
}


if(mype==8)
{
FILE *data8=fopen("lax8.txt", "w");
if(data8==NULL)
    {
    printf("Error opening file!\n");
    return 1;  // Indicate failure
    }

for(int i = 0; i<N/10; i++)
    for(int j=0; j<N/4; j++)
        c8[i][j] = 1;


//write initial data to file
for(int i = 0; i<N/4; i++)
    { 
    for(int j=0; j<N/4; j++)  
        fprintf(data8,"%.4f ",c8[i][j]);
    fprintf(data8,"\n");
    }

t17 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #1 and #2
    for(int j=0;j<N/4;j++)
        {
        send8to4[j]=c8[0][j];
        send8to9[j]=c8[j][N/4-1];
        send8to11[j]=c8[j][0];
        send8to12[j]=c8[N/4-1][j];
        }

    //MPI COMM with #4
    MPI_Send(send8to4,N/4,MPI_DOUBLE,4,n,MPI_COMM_WORLD);
    MPI_Recv(guard8from4, N/4,MPI_DOUBLE,4,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #9
    MPI_Recv(guard8from9, N/4,MPI_DOUBLE,9,n,MPI_COMM_WORLD,&status);
    MPI_Send(send8to9,N/4,MPI_DOUBLE,9,n,MPI_COMM_WORLD);

    //MPI COMM with #11
    MPI_Recv(guard8from11, N/4,MPI_DOUBLE,11,n,MPI_COMM_WORLD,&status);
    MPI_Send(send8to11,N/4,MPI_DOUBLE,11,n,MPI_COMM_WORLD);

    //MPI COMM with #12
    MPI_Recv(guard8from12, N/4,MPI_DOUBLE,12,n,MPI_COMM_WORLD,&status);
    MPI_Send(send8to12,N/4,MPI_DOUBLE,12,n,MPI_COMM_WORLD);


    //compute c4_new[0][j].for j=1,2,..,N/2-2, c2[N/2-1][*]
    //c4[-1][j] = guard0from2[N/2 + j]
    x= -L/2.0 + (N/2 + 0 )*dx;
    v=(-1.414)*x;
    for(int j=1;j<N/4-1;j++)
    {
    y = -L/2.0 + j*dx;
    u = 1.414 * y;
    c8_new[0][j]=.25*(guard8from4[j] + c8[1][j] + c8[0][j-1] + c8[0][j+1]) 
    - dt/(2*dx) * (u * (c8[1][j]-guard8from4[j]) + v * (c8[0][j+1]-c8[0][j-1]));
    }

    //compute c4_new[0][0] , need to use c[-1][0] = guard0from2[N/2] and c[0][-1]=guard0from1[N/2]
    y = -L/2.0;
    u = 1.414 * y;
    c8_new[0][0]=.25*(guard8from4[0] + c8[1][0] + guard8from11[0] + c8[0][1])
    - dt/(2*dx) * (u* (c8[1][0]-guard8from4[0]) + v* (c8[0][1]-guard8from11[0]));

    //compute c4_new[0][N/4-1], need to use c[0][N/2]=guard0from1[0], c[-1][N/2-1]=guard0from2[N-1]
    y = -L/2.0 + dx * (N/4-1);
    u = 1.414 * y;
    c8_new[0][N/4-1]=.25*(guard8from4[N/4-1] + c8[1][N/4-1] + c8[0][N/4-2] + guard8from9[0])
    - dt/(2*dx) * ((u*(c8[1][N/4-1] - guard8from4[N/4-1])) + v* (guard8from9[0] - c8[0][N/4-2]));

    #pragma omp parallel for default(shared) schedule(static) num_threads(nt)
    //compute c8_new from row 1 to row N/4-2
    for(int i=1;i<N/4-1;i++)
       { 
        x= -L/2.0 + (N/2 + i)*dx;
        v= -1.414 * x;

        //compute c_new[i][0]
        y= -L/2; 
        u= 1.414*y;
        c8_new[i][0]=.25*(c8[i-1][0] + c8[i+1][0] + guard8from11[i] + c8[i][1])
        - dt/(2*dx) * (u * (c8[i+1][0]-c8[i-1][0]) + v * (c8[i][1]-guard8from11[i]));
        //compute c8_new[i][j] for j=1,2..,N/4-2.
        for(int j=1;j<N/4-1;j++)
            {
            y = -L/2.0 + j*dx;
            u = 1.414 * y;
            c8_new[i][j]=.25*(c8[i-1][j] + c8[i+1][j] + c8[i][j-1] + c8[i][j+1]) 
            - dt/(2*dx) * (u * (c8[i+1][j]-c8[i-1][j]) + v * (c8[i][j+1]-c8[i][j-1]));
            }

        //compute c8_new[i][N/4-1]
        y=-L/2 + (N/4-1)*dx;
        u = 1.414*y;
        c8_new[i][N/4-1]=.25*(c8[i-1][N/4-1] + c8[i+1][N/4-1] + c8[i][N/4-2] + guard8from9[i]) 
        - dt/(2*dx) * (u * (c8[i+1][N/4-1]-c8[i-1][N/4-1]) + v * (guard8from9[i]-c8[i][N/4-2]));

        }

        //compute c8_new[N/4-1][j]for j = 1,2..N/2-2
        x = -L/2 + (3*N/4 - 1)*dx;
        v = -1.414 * x;
        for(int j=1; j<N/4-1; j++)
            {
            y = -L/2.0 + j*dx;
            u = 1.414 * y;
            c8_new[N/4-1][j]=.25*(c8[N/4 - 2][j] + guard8from12[j] + c8[N/4 - 1][j-1] + c8[N/4 - 1][j+1])
            -dt/(2*dx) * (u * (guard8from12[j]-c8[N/4 - 2][j]) + v * (c8[N/4 - 1][j+1]-c8[N/4 - 1][j-1]));
            }
        //compute c8_new[N/4-1][0], c0[N/2][0]=guard0from2[0], c0[N/2-1][-1]=guard0from1[N-1]
        y = -L/2.0;
        u = 1.414 * y;
        c8_new[N/4-1][0]=.25*(c8[N/4 - 2][0] + guard8from12[0] + guard8from11[N/4-1] + c8[N/4 - 1][1])
        -dt/(2*dx) * (u * (guard8from12[0]-c8[N/4 - 2][0]) + v * (c8[N/4 - 1][1]-guard8from11[N/4-1]));

        //c0[N/2-1][N/2] = guard0from1[N/2-1], c0[N/2][N/2-1]=guard0from2[N/2-1]
        y = -L/2.0 + (N/4-1)*dx;
        u = 1.414 * y;
        c8_new[N/4-1][N/4-1]=.25*(c8[N/4 - 2][N/4-1] + guard8from12[N/4-1] + c8[N/4-1][N/4-2] + guard8from9[N/4-1])
        -dt/(2*dx) * (u * (guard8from12[N/4-1]-c8[N/4 - 2][N/4-1]) + v * (guard8from9[N/4-1]-c8[N/4-1][N/4-2]));

    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/2;i++)
            {
            for(int j=0;j<N/2;j++)
                fprintf(data8,"%.4f ",c8_new[i][j]); 
            fprintf(data8,"\n"); 
            }
       } 

    pt8=c8;
    c8=c8_new;
    c8_new=pt8;
    }
t18 = omp_get_wtime();
fclose(data8);
}

else if(mype==9)
{
FILE *data9=fopen("lax9.txt", "w");
if(data9==NULL)
    {
    printf("Error opening file!\n");
    return 1;  // Indicate failure
    }

for(int i = 0; i<N/10; i++)
    for(int j=0; j<N/4; j++)
        c9[i][j] = 1;

//write initial data to file
for(int i = 0; i<N/4; i++)
    { 
    for(int j=0; j<N/4; j++)  
        fprintf(data9,"%.4f ",c9[i][j]);
    fprintf(data9,"\n");
    }

t19 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #1 and #2
    for(int j=0;j<N/4;j++)
        {
        send9to5[j]=c9[0][j];
        send9to8[j]=c9[j][0];
        send9to10[j]=c9[j][N/4-1];
        send9to13[j]=c9[N/4-1][j];
        }

    //MPI COMM with #5
    MPI_Send(send9to5,N/4,MPI_DOUBLE,5,n,MPI_COMM_WORLD);
    MPI_Recv(guard9from5, N/4,MPI_DOUBLE,5,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #8
    MPI_Send(send9to8,N/4,MPI_DOUBLE,8,n,MPI_COMM_WORLD);
    MPI_Recv(guard9from8, N/4,MPI_DOUBLE,8,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #10
    MPI_Recv(guard9from10, N/4,MPI_DOUBLE,10,n,MPI_COMM_WORLD,&status);
    MPI_Send(send9to10,N/4,MPI_DOUBLE,10,n,MPI_COMM_WORLD);

    //MPI COMM with #13
    MPI_Recv(guard9from13, N/4,MPI_DOUBLE,13,n,MPI_COMM_WORLD,&status);
    MPI_Send(send9to13,N/4,MPI_DOUBLE,13,n,MPI_COMM_WORLD);


    //compute c5_new[0][j].for j=1,2,..,N/2-2, c2[N/2-1][*]
    //c5[-1][j] = guard0from2[N/2 + j]
    x= -L/2 + dx *(N/2 + 0);
    v=(-1.414)*x;
    for(int j=1;j<N/4-1;j++)
        {
        y = -L/2.0 + (N/4+j)*dx;
        u = 1.414 * y;
        c9_new[0][j]=.25*(guard9from5[j] + c9[1][j] + c9[0][j-1] + c9[0][j+1]) 
        - dt/(2*dx) * (u * (c9[1][j]-guard9from5[j]) + v * (c9[0][j+1]-c9[0][j-1]));
        }

    //compute c9_new[0][0] , need to use c[-1][0] = guard0from2[N/2] and c[0][-1]=guard0from1[N/2]
    y = -L/2.0 + (N/4)*dx;
    u = 1.414 * y;
    c9_new[0][0]=.25*(guard9from5[0] + c9[1][0] + guard9from8[0] + c9[0][1])
    - dt/(2*dx) * (u* (c9[1][0]-guard9from5[0]) + v* (c9[0][1]-guard9from8[0]));


    //compute c5_new[0][N/4-1], need to use c[0][N/2]=guard0from1[0], c[-1][N/2-1]=guard0from2[N-1]
    y = -L/2.0 + dx * (N/2-1);
    u = 1.414 * y;
    c9_new[0][N/4-1]=.25*(guard9from5[N/4-1] + c9[1][N/4-1] + c9[0][N/4-2] + guard9from10[0])
    - dt/(2*dx) * ((u*(c9[1][N/4-1] - guard9from5[N/4-1])) + v* (guard9from10[0] - c9[0][N/4-2]));

    #pragma omp parallel for default(shared) schedule(static) num_threads(nt)
    //compute c9_new from row 1 to row N/4-2
    for(int i=1;i<N/4-1;i++)
       { 
        x= -L/2.0 + (N/2+i)*dx;
        v= -1.414 * x;

        //compute c5_new[i][0]
        y= -L/2 + (N/4)*dx; 
        u= 1.414*y;
        c9_new[i][0]=.25*(c9[i-1][0] + c9[i+1][0] + guard9from8[i] + c9[i][1])
        - dt/(2*dx) * (u * (c9[i+1][0]-c9[i-1][0]) + v * (c9[i][1]-guard9from8[i]));
        //compute c9_new[i][j] for j=1,2..,N/4-2.
        for(int j=1;j<N/4-1;j++)
            {
            y = -L/2.0 + (N/4+j)*dx;
            u = 1.414 * y;
            c9_new[i][j]=.25*(c9[i-1][j] + c9[i+1][j] + c9[i][j-1] + c9[i][j+1]) 
            - dt/(2*dx) * (u * (c9[i+1][j]-c9[i-1][j]) + v * (c9[i][j+1]-c9[i][j-1]));
            }

        //compute c9_new[i][N/4-1]
        y=-L/2 + (N/2-1)*dx;
        u = 1.414*y;
        c9_new[i][N/4-1]=.25*(c9[i-1][N/4-1] + c9[i+1][N/4-1] + c9[i][N/4-2] + guard9from10[i]) 
        - dt/(2*dx) * (u * (c9[i+1][N/4-1]-c9[i-1][N/4-1]) + v * (guard9from10[i]-c9[i][N/4-2]));
        }

        //compute c9_new[N/4-1][*]. Need to use c[N/2][j] = guard0from2[j] for j = 1,2..N/2-2
        x = -L/2 + (3*N/4 - 1)*dx;
        v = -1.414 * x;
        for(int j=1; j<N/4-1; j++)
            {
            y = -L/2.0 + (N/4 + j)*dx;
            u = 1.414 * y;
            c9_new[N/4-1][j]=.25*(c9[N/4 - 2][j] + guard9from13[j] + c9[N/4 - 1][j-1] + c9[N/4 - 1][j+1])
            -dt/(2*dx) * (u * (guard9from13[j]-c9[N/4 - 2][j]) + v * (c9[N/4 - 1][j+1]-c9[N/4 - 1][j-1]));
            }
        //compute c9_new[N/4-1][0], c0[N/2][0]=guard0from2[0], c0[N/2-1][-1]=guard0from1[N-1]
        y = -L/2.0 + (N/4)*dx;
        u = 1.414 * y;
        c9_new[N/4-1][0]=.25*(c9[N/4 - 2][0] + guard9from13[0] + guard9from8[N/4-1] + c9[N/4 - 1][1])
        -dt/(2*dx) * (u * (guard9from13[0]-c9[N/4 - 2][0]) + v * (c9[N/4 - 1][1]-guard9from8[N/4-1]));

        //c1[N/4-1][N/4] = guard0from1[N/2-1], c0[N/2][N/2-1]=guard0from2[N/2-1]
        y = -L/2.0 + (N/2-1)*dx;
        u = 1.414 * y;
        c9_new[N/4-1][N/4-1]=.25*(c9[N/4 - 2][N/4-1] + guard9from13[N/4-1] + c9[N/4-1][N/4-2] + guard9from10[N/4-1])
        -dt/(2*dx) * (u * (guard9from13[N/4-1]-c9[N/4 - 2][N/4-1]) + v * (guard9from10[N/4-1]-c9[N/4-1][N/4-2]));

    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/4;i++)
            {
            for(int j=0;j<N/4;j++)
                fprintf(data9,"%.4f ",c9_new[i][j]); 
            fprintf(data9,"\n"); 
            }
       } 

    pt9=c9;
    c9=c9_new;
    c9_new=pt9;
    }
t20 = omp_get_wtime();
fclose(data9);
}


else if(mype==10)
{
FILE *data10=fopen("lax10.txt", "w");
if(data10==NULL)
    {
    printf("Error opening file!\n");
    return 1;  // Indicate failure
    }


for(int i = 0; i<N/10; i++)
    for(int j=0; j<N/4; j++)
        c10[i][j] = 1;

//write initial data to file
for(int i = 0; i<N/4; i++)
    { 
    for(int j=0; j<N/4; j++)  
        fprintf(data10,"%.4f ",c10[i][j]);
    fprintf(data10,"\n");
    }

t21 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #1 and #2
    for(int j=0;j<N/4;j++)
        {
        send10to6[j]=c10[0][j];
        send10to9[j]=c10[j][0];
        send10to11[j]=c10[j][N/4-1];
        send10to14[j]=c10[N/4-1][j];
        }

    //MPI COMM with #6
    MPI_Send(send10to6,N/4,MPI_DOUBLE,6,n,MPI_COMM_WORLD);
    MPI_Recv(guard10from6, N/4,MPI_DOUBLE,6,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #9
    MPI_Send(send10to9,N/4,MPI_DOUBLE,9,n,MPI_COMM_WORLD);
    MPI_Recv(guard10from9, N/4,MPI_DOUBLE,9,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #11
    MPI_Recv(guard10from11, N/4,MPI_DOUBLE,11,n,MPI_COMM_WORLD,&status);
    MPI_Send(send10to11,N/4,MPI_DOUBLE,11,n,MPI_COMM_WORLD);

    //MPI COMM with #14
    MPI_Recv(guard10from14, N/4,MPI_DOUBLE,14,n,MPI_COMM_WORLD,&status);
    MPI_Send(send10to14,N/4,MPI_DOUBLE,14,n,MPI_COMM_WORLD);


    //compute c10_new[0][j].for j=1,2,..,N/2-2, c2[N/2-1][*]
    //c0[-1][j] = guard0from2[N/2 + j]
    x= -L/2 + (N/2 + 0)*dx;
    v=(-1.414)*x;
    for(int j=1;j<N/4-1;j++)
        {
        y = -L/2.0 + (N/2+j)*dx;
        u = 1.414 * y;
        c10_new[0][j]=.25*(guard10from6[j] + c10[1][j] + c10[0][j-1] + c10[0][j+1]) 
        - dt/(2*dx) * (u * (c10[1][j]-guard10from6[j]) + v * (c10[0][j+1]-c10[0][j-1]));
        }

    //compute c10_new[0][0] , need to use c[-1][0] = guard0from2[N/2] and c[0][-1]=guard0from1[N/2]
    y = -L/2.0 + (N/2)*dx;
    u = 1.414 * y;
    c10_new[0][0]=.25*(guard10from6[0] + c10[1][0] + guard10from9[0] + c10[0][1])
    - dt/(2*dx) * (u* (c10[1][0]-guard10from6[0]) + v* (c10[0][1]-guard10from9[0]));


    //compute c10_new[0][N/4-1], need to use c[0][N/2]=guard0from1[0], c[-1][N/2-1]=guard0from2[N-1]
    y = -L/2.0 + dx * (3*N/4-1);
    u = 1.414 * y;
    c10_new[0][N/4-1]=.25*(guard10from6[N/4-1] + c10[1][N/4-1] + c10[0][N/4-2] + guard10from11[0])
    - dt/(2*dx) * ((u*(c10[1][N/4-1] - guard10from6[N/4-1])) + v* (guard10from11[0] - c10[0][N/4-2]));

    #pragma omp parallel for default(shared) schedule(static) num_threads(nt)
    //compute c10_new from row 1 to row N/4-2
    for(int i=1;i<N/4-1;i++)
       { 
        x= -L/2.0 + (N/2+i)*dx;
        v= -1.414 * x;
        //compute c_new[i][0]
        y= -L/2 + (N/2)*dx; 
        u= 1.414*y;
        c10_new[i][0]=.25*(c10[i-1][0] + c10[i+1][0] + guard10from9[i] + c10[i][1])
        - dt/(2*dx) * (u * (c10[i+1][0]-c10[i-1][0]) + v * (c10[i][1]-guard10from9[i]));
        //compute c10_new[i][j] for j=1,2..,N/4-2.
        for(int j=1;j<N/4-1;j++)
            {
            y = -L/2.0 + (N/2+j)*dx;
            u = 1.414 * y;
            c10_new[i][j]=.25*(c10[i-1][j] + c10[i+1][j] + c10[i][j-1] + c10[i][j+1]) 
            - dt/(2*dx) * (u * (c10[i+1][j]-c10[i-1][j]) + v * (c10[i][j+1]-c10[i][j-1]));
            }

        //compute c10_new[i][N/4-1]
        y=-L/2 + (3*N/4-1)*dx;
        u = 1.414*y;
        c10_new[i][N/4-1]=.25*(c10[i-1][N/4-1] + c10[i+1][N/4-1] + c10[i][N/4-2] + guard10from11[i]) 
        - dt/(2*dx) * (u * (c10[i+1][N/4-1]-c10[i-1][N/4-1]) + v * (guard10from11[i]-c10[i][N/4-2]));

        }

        //compute c6_new[N/4-1][*]. Need to use c[N/2][j] = guard0from2[j] for j = 1,2..N/2-2
        x = -L/2 + (3*N/4 - 1)*dx;
        v = -1.414 * x;
        for(int j=1; j<N/4-1; j++)
            {
            y = -L/2.0 + (N/2 + j)*dx;
            u = 1.414 * y;
            c10_new[N/4-1][j]=.25*(c10[N/4 - 2][j] + guard10from14[j] + c10[N/4 - 1][j-1] + c10[N/4 - 1][j+1])
            -dt/(2*dx) * (u * (guard10from14[j]-c10[N/4 - 2][j]) + v * (c10[N/4 - 1][j+1]-c10[N/4 - 1][j-1]));
            }
        //compute c6_new[N/4-1][0], c0[N/2][0]=guard0from2[0], c0[N/2-1][-1]=guard0from1[N-1]
        y = -L/2.0 + (N/2)*dx;
        u = 1.414 * y;
        c10_new[N/4-1][0]=.25*(c10[N/4 - 2][0] + guard10from14[0] + guard10from9[N/4-1] + c10[N/4 - 1][1])
        -dt/(2*dx) * (u * (guard10from14[0]-c10[N/4 - 2][0]) + v * (c10[N/4 - 1][1]-guard10from9[N/4-1]));

        //c1[N/4-1][N/4] = guard0from1[N/2-1], c0[N/2][N/2-1]=guard0from2[N/2-1]
        y = -L/2.0 + (3*N/4-1)*dx;
        u = 1.414 * y;
        c10_new[N/4-1][N/4-1]=.25*(c10[N/4 - 2][N/4-1] + guard10from14[N/4-1] + c10[N/4-1][N/4-2] + guard10from11[N/4-1])
        -dt/(2*dx) * (u * (guard10from14[N/4-1]-c10[N/4 - 2][N/4-1]) + v * (guard10from11[N/4-1]-c10[N/4-1][N/4-2]));

    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/4;i++)
            {
            for(int j=0;j<N/4;j++)
                fprintf(data10,"%.4f ",c10_new[i][j]); 
            fprintf(data10,"\n"); 
            }
       } 

    pt10=c10;
    c10=c10_new;
    c10_new=pt10;
    }
t22 = omp_get_wtime();
fclose(data10);
}


else if(mype==11)
{
FILE *data11=fopen("lax11.txt", "w");
if(data11==NULL)
    {
    printf("Error opening file!\n");
    return 1;  // Indicate failure
    }

for(int i = 0; i<N/10; i++)
    for(int j=0; j<N/4; j++)
        c11[i][j] = 1;

//write initial data to file
for(int i = 0; i<N/4; i++)
    { 
    for(int j=0; j<N/4; j++)  
        fprintf(data11,"%.4f ",c11[i][j]);
    fprintf(data11,"\n");
    }

t23 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #1 and #2
    for(int j=0;j<N/4;j++)
        {
        send11to7[j]=c11[0][j];
        send11to8[j]=c11[j][N/4-1];
        send11to10[j]=c11[j][0];
        send11to15[j]=c11[N/4-1][j];
        }

    //MPI COMM with #7
    MPI_Send(send11to7,N/4,MPI_DOUBLE,7,n,MPI_COMM_WORLD);
    MPI_Recv(guard11from7, N/4,MPI_DOUBLE,7,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #8
    MPI_Send(send11to8,N/4,MPI_DOUBLE,8,n,MPI_COMM_WORLD);
    MPI_Recv(guard11from8, N/4,MPI_DOUBLE,8,n,MPI_COMM_WORLD,&status);
 

    //MPI COMM with #10
    MPI_Send(send11to10,N/4,MPI_DOUBLE,10,n,MPI_COMM_WORLD);
    MPI_Recv(guard11from10, N/4,MPI_DOUBLE,10,n,MPI_COMM_WORLD,&status);

 
    //MPI COMM with #15
    MPI_Recv(guard11from15, N/4,MPI_DOUBLE,15,n,MPI_COMM_WORLD,&status);
    MPI_Send(send11to15, N/4,MPI_DOUBLE,15,n,MPI_COMM_WORLD);


    //compute c11_new[0][j].for j=1,2,..,N/2-2, c2[N/2-1][*]
    //c0[-1][j] = guard0from2[N/2 + j]
    x= -L/2 + (N/2) * dx;
    v=(-1.414)*x;
    for(int j=1;j<N/4-1;j++)
        {
        y = -L/2.0 + (3*N/4 + j)*dx;
        u = 1.414 * y;
        c11_new[0][j]=.25*(guard11from7[j] + c11[1][j] + c11[0][j-1] + c11[0][j+1]) 
        - dt/(2*dx) * (u * (c11[1][j]-guard11from7[j]) + v * (c11[0][j+1]-c11[0][j-1]));
        }

    //compute c11_new[0][0] , need to use c[-1][0] = guard0from2[N/2] and c[0][-1]=guard0from1[N/2]
    y = -L/2.0 + (3*N/4)*dx;
    u = 1.414 * y;
    c11_new[0][0]=.25*(guard11from7[0] + c11[1][0] + guard11from10[0] + c11[0][1])
    - dt/(2*dx) * (u* (c11[1][0]-guard11from7[0]) + v* (c11[0][1]-guard11from10[0]));


    //compute c11_new[0][N/4-1], need to use c[0][N/2]=guard0from1[0], c[-1][N/2-1]=guard0from2[N-1]
    y = -L/2.0 + dx * (N-1);
    u = 1.414 * y;
    c11_new[0][N/4-1]=.25*(guard11from7[N/4-1] + c11[1][N/4-1] + c11[0][N/4-2] + guard11from8[0])
    - dt/(2*dx) * ((u*(c11[1][N/4-1] - guard11from7[N/4-1])) + v* (guard11from8[0] - c11[0][N/4-2]));


    #pragma omp parallel for default(shared) schedule(static) num_threads(nt)
    //compute c11_new from row 1 to row N/4-2
    for(int i=1;i<N/4-1;i++)
       { 
        x= -L/2.0 + (N/2+i)*dx;
        v= -1.414 * x;

        //compute c3_new[i][0]
        y= -L/2 + (3*N/4)*dx; 
        u= 1.414*y;
        c11_new[i][0]=.25*(c11[i-1][0] + c11[i+1][0] + guard11from10[i] + c11[i][1])
        - dt/(2*dx) * (u * (c11[i+1][0]-c11[i-1][0]) + v * (c11[i][1]-guard11from10[i]));
        //compute c3_new[i][j] for j=1,2..,N/4-2.
        for(int j=1;j<N/4-1;j++)
            {
            y = -L/2.0 + (3*N/4+j)*dx;
            u = 1.414 * y;
            c11_new[i][j]=.25*(c11[i-1][j] + c11[i+1][j] + c11[i][j-1] + c11[i][j+1]) 
            - dt/(2*dx) * (u * (c11[i+1][j]-c11[i-1][j]) + v * (c11[i][j+1]-c11[i][j-1]));
            }

        //compute c3_new[i][N/4-1]
        y=-L/2 + (N-1)*dx;
        u = 1.414*y;
        c11_new[i][N/4-1]=.25*(c11[i-1][N/4-1] + c11[i+1][N/4-1] + c11[i][N/4-2] + guard11from8[i]) 
        - dt/(2*dx) * (u * (c11[i+1][N/4-1]-c11[i-1][N/4-1]) + v * (guard11from8[i]-c11[i][N/4-2]));

        }

        //compute c11_new[N/4-1][*]. Need to use c[N/2][j] = guard0from2[j] for j = 1,2..N/2-2
        x = -L/2 + (3*N/4 - 1)*dx;
        v = -1.414 * x;
        for(int j=1; j<N/4-1; j++)
            {
            y = -L/2.0 + (3*N/4 + j)*dx;
            u = 1.414 * y;
            c11_new[N/4-1][j]=.25*(c11[N/4 - 2][j] + guard11from15[j] + c11[N/4 - 1][j-1] + c11[N/4 - 1][j+1])
            -dt/(2*dx) * (u * (guard11from15[j]-c11[N/4 - 2][j]) + v * (c11[N/4 - 1][j+1]-c11[N/4 - 1][j-1]));
            }
        //compute c11_new[N/4-1][0], c0[N/2][0]=guard0from2[0], c0[N/2-1][-1]=guard0from1[N-1]
        y = -L/2.0 + (3*N/4)*dx;
        u = 1.414 * y;
        c11_new[N/4-1][0]=.25*(c11[N/4 - 2][0] + guard11from15[0] + guard11from10[N/4-1] + c11[N/4 - 1][1])
        -dt/(2*dx) * (u * (guard11from15[0]-c11[N/4 - 2][0]) + v * (c11[N/4 - 1][1]-guard11from10[N/4-1]));

        //c1[N/4-1][N/4] = guard0from1[N/2-1], c0[N/2][N/2-1]=guard0from2[N/2-1]
        y = -L/2.0 + (N-1)*dx;
        u = 1.414 * y;
        c11_new[N/4-1][N/4-1]=.25*(c11[N/4 - 2][N/4-1] + guard11from15[N/4-1] + c11[N/4-1][N/4-2] + guard11from8[N/4-1])
        -dt/(2*dx) * (u * (guard11from15[N/4-1]-c11[N/4 - 2][N/4-1]) + v * (guard11from8[N/4-1]-c11[N/4-1][N/4-2]));

    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/4;i++)
            {
            for(int j=0;j<N/4;j++)
                fprintf(data11,"%.4f ",c11_new[i][j]); 
            fprintf(data11,"\n"); 
            }
       } 

    pt11=c11;
    c11=c11_new;
    c11_new=pt11;
    }
t24 = omp_get_wtime();
fclose(data11);
}


if(mype==12)
{
FILE *data12=fopen("lax12.txt", "w");
if(data12==NULL)
    {
    printf("Error opening file!\n");
    return 1;  // Indicate failure
    }

//write initial data to file
for(int i = 0; i<N/4; i++)
    { 
    for(int j=0; j<N/4; j++)  
        fprintf(data12,"%.4f ",c12[i][j]);
    fprintf(data12,"\n");
    }

t25 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #1 and #2
    for(int j=0;j<N/4;j++)
        {
        send12to0[j]=c12[N/4-1][j];
        send12to8[j]=c12[0][j];
        send12to13[j]=c12[j][N/4-1];
        send12to15[j]=c12[j][0];
        }

    //MPI COMM with #0
    MPI_Send(send12to0,N/4,MPI_DOUBLE,0,n,MPI_COMM_WORLD);
    MPI_Recv(guard12from0, N/4,MPI_DOUBLE,0,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #8
    MPI_Send(send12to8,N/4,MPI_DOUBLE,8,n,MPI_COMM_WORLD);
    MPI_Recv(guard12from8, N/4,MPI_DOUBLE,8,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #13
    MPI_Recv(guard12from13, N/4,MPI_DOUBLE,13,n,MPI_COMM_WORLD,&status);
    MPI_Send(send12to13,N/4,MPI_DOUBLE,13,n,MPI_COMM_WORLD);

    //MPI COMM with #15
    MPI_Recv(guard12from15, N/4,MPI_DOUBLE,15,n,MPI_COMM_WORLD,&status);
    MPI_Send(send12to15,N/4,MPI_DOUBLE,15,n,MPI_COMM_WORLD);


    //compute c4_new[0][j].for j=1,2,..,N/2-2, c2[N/2-1][*]
    //c12[-1][j] = guard0from2[N/2 + j]
    x= -L/2.0 + (3*N/4 + 0 )*dx;
    v=(-1.414)*x;
    for(int j=1;j<N/4-1;j++)
    {
    y = -L/2.0 + j*dx;
    u = 1.414 * y;
    c12_new[0][j]=.25*(guard12from8[j] + c12[1][j] + c12[0][j-1] + c12[0][j+1]) 
    - dt/(2*dx) * (u * (c12[1][j]-guard12from8[j]) + v * (c12[0][j+1]-c12[0][j-1]));
    }

    //compute c4_new[0][0] , need to use c[-1][0] = guard0from2[N/2] and c[0][-1]=guard0from1[N/2]
    y = -L/2.0;
    u = 1.414 * y;
    c12_new[0][0]=.25*(guard12from8[0] + c12[1][0] + guard12from15[0] + c12[0][1])
    - dt/(2*dx) * (u* (c12[1][0]-guard12from8[0]) + v* (c12[0][1]-guard12from15[0]));

    //compute c4_new[0][N/4-1], need to use c[0][N/2]=guard0from1[0], c[-1][N/2-1]=guard0from2[N-1]
    y = -L/2.0 + dx * (N/4-1);
    u = 1.414 * y;
    c12_new[0][N/4-1]=.25*(guard12from8[N/4-1] + c12[1][N/4-1] + c12[0][N/4-2] + guard12from13[0])
    - dt/(2*dx) * ((u*(c12[1][N/4-1] - guard12from8[N/4-1])) + v* (guard12from13[0] - c12[0][N/4-2]));


    #pragma omp parallel for default(shared) schedule(static) num_threads(nt)
    //compute c12_new from row 1 to row N/4-2
    for(int i=1;i<N/4-1;i++)
       { 
        x= -L/2.0 + (3*N/4 + i)*dx;
        v= -1.414 * x;

        //compute c_new[i][0]
        y= -L/2; 
        u= 1.414*y;
        c12_new[i][0]=.25*(c12[i-1][0] + c12[i+1][0] + guard12from15[i] + c12[i][1])
        - dt/(2*dx) * (u * (c12[i+1][0]-c12[i-1][0]) + v * (c12[i][1]-guard12from15[i]));
        //compute c12_new[i][j] for j=1,2..,N/4-2.
        for(int j=1;j<N/4-1;j++)
            {
            y = -L/2.0 + j*dx;
            u = 1.414 * y;
            c12_new[i][j]=.25*(c12[i-1][j] + c12[i+1][j] + c12[i][j-1] + c12[i][j+1]) 
            - dt/(2*dx) * (u * (c12[i+1][j]-c12[i-1][j]) + v * (c12[i][j+1]-c12[i][j-1]));
            }

        //compute c12_new[i][N/4-1]
        y=-L/2 + (N/4-1)*dx;
        u = 1.414*y;
        c12_new[i][N/4-1]=.25*(c12[i-1][N/4-1] + c12[i+1][N/4-1] + c12[i][N/4-2] + guard12from13[i]) 
        - dt/(2*dx) * (u * (c12[i+1][N/4-1]-c12[i-1][N/4-1]) + v * (guard12from13[i]-c12[i][N/4-2]));

        }

        //compute c12_new[N/4-1][j]for j = 1,2..N/2-2
        x = -L/2 + (N - 1)*dx;
        v = -1.414 * x;
        for(int j=1; j<N/4-1; j++)
            {
            y = -L/2.0 + j*dx;
            u = 1.414 * y;
            c12_new[N/4-1][j]=.25*(c12[N/4 - 2][j] + guard12from0[j] + c12[N/4 - 1][j-1] + c12[N/4 - 1][j+1])
            -dt/(2*dx) * (u * (guard12from0[j]-c12[N/4 - 2][j]) + v * (c12[N/4 - 1][j+1]-c12[N/4 - 1][j-1]));
            }
        //compute c12_new[N/4-1][0], c0[N/2][0]=guard0from2[0], c0[N/2-1][-1]=guard0from1[N-1]
        y = -L/2.0;
        u = 1.414 * y;
        c12_new[N/4-1][0]=.25*(c12[N/4 - 2][0] + guard12from0[0] + guard12from15[N/4-1] + c12[N/4 - 1][1])
        -dt/(2*dx) * (u * (guard12from0[0]-c12[N/4 - 2][0]) + v * (c12[N/4 - 1][1]-guard12from15[N/4-1]));

        //c0[N/2-1][N/2] = guard0from1[N/2-1], c0[N/2][N/2-1]=guard0from2[N/2-1]
        y = -L/2.0 + (N/4-1)*dx;
        u = 1.414 * y;
        c12_new[N/4-1][N/4-1]=.25*(c12[N/4 - 2][N/4-1] + guard12from0[N/4-1] + c12[N/4-1][N/4-2] + guard12from13[N/4-1])
        -dt/(2*dx) * (u * (guard12from0[N/4-1]-c12[N/4 - 2][N/4-1]) + v * (guard12from13[N/4-1]-c12[N/4-1][N/4-2]));

    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/2;i++)
            {
            for(int j=0;j<N/2;j++)
                fprintf(data12,"%.4f ",c12_new[i][j]); 
            fprintf(data12,"\n"); 
            }
       } 

    pt12=c12;
    c12=c12_new;
    c12_new=pt12;
    }
t26 = omp_get_wtime();
fclose(data12);
}

else if(mype==13)
{
FILE *data13=fopen("lax13.txt", "w");
if(data13==NULL)
    {
    printf("Error opening file!\n");
    return 1;  // Indicate failure
    }

//write initial data to file
for(int i = 0; i<N/4; i++)
    { 
    for(int j=0; j<N/4; j++)  
        fprintf(data13,"%.4f ",c13[i][j]);
    fprintf(data13,"\n");
    }

t27 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #1 and #2
    for(int j=0;j<N/4;j++)
        {
        send13to1[j]=c13[N/4-1][j];
        send13to9[j]=c13[0][j];
        send13to12[j]=c13[j][0];
        send13to14[j]=c13[j][N/4-1];
        }

    //MPI COMM with #1
    MPI_Send(send13to1,N/4,MPI_DOUBLE,1,n,MPI_COMM_WORLD);
    MPI_Recv(guard13from1, N/4,MPI_DOUBLE,1,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #9
    MPI_Send(send13to9,N/4,MPI_DOUBLE,9,n,MPI_COMM_WORLD);
    MPI_Recv(guard13from9, N/4,MPI_DOUBLE,9,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #12
    MPI_Send(send13to12,N/4,MPI_DOUBLE,12,n,MPI_COMM_WORLD);
    MPI_Recv(guard13from12, N/4,MPI_DOUBLE,12,n,MPI_COMM_WORLD,&status);


    //MPI COMM with #14
    MPI_Recv(guard13from14, N/4,MPI_DOUBLE,14,n,MPI_COMM_WORLD,&status);
    MPI_Send(send13to14,N/4,MPI_DOUBLE,14,n,MPI_COMM_WORLD);


    //compute c5_new[0][j].for j=1,2,..,N/2-2, c2[N/2-1][*]
    //c5[-1][j] = guard0from2[N/2 + j]
    x= -L/2 + dx *(3*N/4 + 0);
    v=(-1.414)*x;
    for(int j=1;j<N/4-1;j++)
        {
        y = -L/2.0 + (N/4+j)*dx;
        u = 1.414 * y;
        c13_new[0][j]=.25*(guard13from9[j] + c13[1][j] + c13[0][j-1] + c13[0][j+1]) 
        - dt/(2*dx) * (u * (c13[1][j]-guard13from9[j]) + v * (c13[0][j+1]-c13[0][j-1]));
        }

    //compute c13_new[0][0] , need to use c[-1][0] = guard0from2[N/2] and c[0][-1]=guard0from1[N/2]
    y = -L/2.0 + (N/4)*dx;
    u = 1.414 * y;
    c13_new[0][0]=.25*(guard13from9[0] + c13[1][0] + guard13from12[0] + c13[0][1])
    - dt/(2*dx) * (u* (c13[1][0]-guard13from9[0]) + v* (c13[0][1]-guard13from12[0]));


    //compute c5_new[0][N/4-1], need to use c[0][N/2]=guard0from1[0], c[-1][N/2-1]=guard0from2[N-1]
    y = -L/2.0 + dx * (N/2-1);
    u = 1.414 * y;
    c13_new[0][N/4-1]=.25*(guard13from9[N/4-1] + c13[1][N/4-1] + c13[0][N/4-2] + guard13from14[0])
    - dt/(2*dx) * ((u*(c13[1][N/4-1] - guard13from9[N/4-1])) + v* (guard13from14[0] - c13[0][N/4-2]));


#pragma omp parallel for default(shared) schedule(static) num_threads(nt)
    //compute c13_new from row 1 to row N/4-2
    for(int i=1;i<N/4-1;i++)
       { 
        x= -L/2.0 + (3*N/4+i)*dx;
        v= -1.414 * x;

        //compute c5_new[i][0]
        y= -L/2 + (N/4)*dx; 
        u= 1.414*y;
        c13_new[i][0]=.25*(c13[i-1][0] + c13[i+1][0] + guard13from12[i] + c13[i][1])
        - dt/(2*dx) * (u * (c13[i+1][0]-c13[i-1][0]) + v * (c13[i][1]-guard13from12[i]));
        //compute c13_new[i][j] for j=1,2..,N/4-2.
        for(int j=1;j<N/4-1;j++)
            {
            y = -L/2.0 + (N/4+j)*dx;
            u = 1.414 * y;
            c13_new[i][j]=.25*(c13[i-1][j] + c13[i+1][j] + c13[i][j-1] + c13[i][j+1]) 
            - dt/(2*dx) * (u * (c13[i+1][j]-c13[i-1][j]) + v * (c13[i][j+1]-c13[i][j-1]));
            }

        //compute c13_new[i][N/4-1]
        y=-L/2 + (N/2-1)*dx;
        u = 1.414*y;
        c13_new[i][N/4-1]=.25*(c13[i-1][N/4-1] + c13[i+1][N/4-1] + c13[i][N/4-2] + guard13from14[i]) 
        - dt/(2*dx) * (u * (c13[i+1][N/4-1]-c13[i-1][N/4-1]) + v * (guard13from14[i]-c13[i][N/4-2]));
        }

        //compute c13_new[N/4-1][*]. Need to use c[N/2][j] = guard0from2[j] for j = 1,2..N/2-2
        x = -L/2 + (N - 1)*dx;
        v = -1.414 * x;
        for(int j=1; j<N/4-1; j++)
            {
            y = -L/2.0 + (N/4 + j)*dx;
            u = 1.414 * y;
            c13_new[N/4-1][j]=.25*(c13[N/4 - 2][j] + guard13from1[j] + c13[N/4 - 1][j-1] + c13[N/4 - 1][j+1])
            -dt/(2*dx) * (u * (guard13from1[j]-c13[N/4 - 2][j]) + v * (c13[N/4 - 1][j+1]-c13[N/4 - 1][j-1]));
            }
        //compute c13_new[N/4-1][0], c0[N/2][0]=guard0from2[0], c0[N/2-1][-1]=guard0from1[N-1]
        y = -L/2.0 + (N/4)*dx;
        u = 1.414 * y;
        c13_new[N/4-1][0]=.25*(c13[N/4 - 2][0] + guard13from1[0] + guard13from12[N/4-1] + c13[N/4 - 1][1])
        -dt/(2*dx) * (u * (guard13from1[0]-c13[N/4 - 2][0]) + v * (c13[N/4 - 1][1]-guard13from12[N/4-1]));

        //c1[N/4-1][N/4] = guard0from1[N/2-1], c0[N/2][N/2-1]=guard0from2[N/2-1]
        y = -L/2.0 + (N/2-1)*dx;
        u = 1.414 * y;
        c13_new[N/4-1][N/4-1]=.25*(c13[N/4 - 2][N/4-1] + guard13from1[N/4-1] + c13[N/4-1][N/4-2] + guard13from14[N/4-1])
        -dt/(2*dx) * (u * (guard13from1[N/4-1]-c13[N/4 - 2][N/4-1]) + v * (guard13from14[N/4-1]-c13[N/4-1][N/4-2]));

    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/4;i++)
            {
            for(int j=0;j<N/4;j++)
                fprintf(data13,"%.4f ",c13_new[i][j]); 
            fprintf(data13,"\n"); 
            }
       } 

    pt13=c13;
    c13=c13_new;
    c13_new=pt13;
    }
t28 = omp_get_wtime();
fclose(data13);
}


else if(mype==14)
{
FILE *data14=fopen("lax14.txt", "w");
if(data14==NULL)
    {
    printf("Error opening file!\n");
    return 1;  // Indicate failure
    }

//write initial data to file
for(int i = 0; i<N/4; i++)
    { 
    for(int j=0; j<N/4; j++)  
        fprintf(data14,"%.4f ",c14[i][j]);
    fprintf(data14,"\n");
    }

t29 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #1 and #2
    for(int j=0;j<N/4;j++)
        {
        send14to2[j]=c14[N/4-1][j];
        send14to10[j]=c14[0][j];
        send14to13[j]=c14[j][0];
        send14to15[j]=c14[j][N/4-1];
        }

    //MPI COMM with #2
    MPI_Send(send14to2,N/4,MPI_DOUBLE,2,n,MPI_COMM_WORLD);
    MPI_Recv(guard14from2, N/4,MPI_DOUBLE,2,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #10
    MPI_Send(send14to10,N/4,MPI_DOUBLE,10,n,MPI_COMM_WORLD);
    MPI_Recv(guard14from10, N/4,MPI_DOUBLE,10,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #13
    MPI_Send(send14to13,N/4,MPI_DOUBLE,13,n,MPI_COMM_WORLD);
    MPI_Recv(guard14from13, N/4,MPI_DOUBLE,13,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #15
    MPI_Recv(guard14from15, N/4,MPI_DOUBLE,15,n,MPI_COMM_WORLD,&status);
    MPI_Send(send14to15,N/4,MPI_DOUBLE,15,n,MPI_COMM_WORLD);


    //compute c14_new[0][j].for j=1,2,..,N/2-2, c2[N/2-1][*]
    //c0[-1][j] = guard0from2[N/2 + j]
    x= -L/2 + (3*N/4 + 0)*dx;
    v=(-1.414)*x;
    for(int j=1;j<N/4-1;j++)
        {
        y = -L/2.0 + (N/2+j)*dx;
        u = 1.414 * y;
        c14_new[0][j]=.25*(guard14from10[j] + c14[1][j] + c14[0][j-1] + c14[0][j+1]) 
        - dt/(2*dx) * (u * (c14[1][j]-guard14from10[j]) + v * (c14[0][j+1]-c14[0][j-1]));
        }

    //compute c14_new[0][0] , need to use c[-1][0] = guard0from2[N/2] and c[0][-1]=guard0from1[N/2]
    y = -L/2.0 + (N/2)*dx;
    u = 1.414 * y;
    c14_new[0][0]=.25*(guard14from10[0] + c14[1][0] + guard14from13[0] + c14[0][1])
    - dt/(2*dx) * (u* (c14[1][0]-guard14from10[0]) + v* (c14[0][1]-guard14from13[0]));


    //compute c14_new[0][N/4-1], need to use c[0][N/2]=guard0from1[0], c[-1][N/2-1]=guard0from2[N-1]
    y = -L/2.0 + dx * (3*N/4-1);
    u = 1.414 * y;
    c14_new[0][N/4-1]=.25*(guard14from10[N/4-1] + c14[1][N/4-1] + c14[0][N/4-2] + guard14from15[0])
    - dt/(2*dx) * ((u*(c14[1][N/4-1] - guard14from10[N/4-1])) + v* (guard14from15[0] - c14[0][N/4-2]));


#pragma omp parallel for default(shared) schedule(static) num_threads(nt)
    //compute c14_new from row 1 to row N/4-2
    for(int i=1;i<N/4-1;i++)
       { 
        x= -L/2.0 + (3*N/4+i)*dx;
        v= -1.414 * x;
        //compute c_new[i][0]
        y= -L/2 + (N/2)*dx; 
        u= 1.414*y;
        c14_new[i][0]=.25*(c14[i-1][0] + c14[i+1][0] + guard14from13[i] + c14[i][1])
        - dt/(2*dx) * (u * (c14[i+1][0]-c14[i-1][0]) + v * (c14[i][1]-guard14from13[i]));
        //compute c14_new[i][j] for j=1,2..,N/4-2.
        for(int j=1;j<N/4-1;j++)
            {
            y = -L/2.0 + (N/2+j)*dx;
            u = 1.414 * y;
            c14_new[i][j]=.25*(c14[i-1][j] + c14[i+1][j] + c14[i][j-1] + c14[i][j+1]) 
            - dt/(2*dx) * (u * (c14[i+1][j]-c14[i-1][j]) + v * (c14[i][j+1]-c14[i][j-1]));
            }

        //compute c14_new[i][N/4-1]
        y=-L/2 + (3*N/4-1)*dx;
        u = 1.414*y;
        c14_new[i][N/4-1]=.25*(c14[i-1][N/4-1] + c14[i+1][N/4-1] + c14[i][N/4-2] + guard14from15[i]) 
        - dt/(2*dx) * (u * (c14[i+1][N/4-1]-c14[i-1][N/4-1]) + v * (guard14from15[i]-c14[i][N/4-2]));

        }

        //compute c6_new[N/4-1][*]. Need to use c[N/2][j] = guard0from2[j] for j = 1,2..N/2-2
        x = -L/2 + (N - 1)*dx;
        v = -1.414 * x;
        for(int j=1; j<N/4-1; j++)
            {
            y = -L/2.0 + (N/2 + j)*dx;
            u = 1.414 * y;
            c14_new[N/4-1][j]=.25*(c14[N/4 - 2][j] + guard14from2[j] + c14[N/4 - 1][j-1] + c14[N/4 - 1][j+1])
            -dt/(2*dx) * (u * (guard14from2[j]-c14[N/4 - 2][j]) + v * (c14[N/4 - 1][j+1] - c14[N/4 - 1][j-1]));
            }
        //compute c6_new[N/4-1][0], c0[N/2][0]=guard0from2[0], c0[N/2-1][-1]=guard0from1[N-1]
        y = -L/2.0 + (N/2)*dx;
        u = 1.414 * y;
        c14_new[N/4-1][0]=.25*(c14[N/4 - 2][0] + guard14from2[0] + guard14from13[N/4-1] + c14[N/4 - 1][1])
        -dt/(2*dx) * (u * (guard14from2[0]-c14[N/4 - 2][0]) + v * (c14[N/4 - 1][1]-guard14from13[N/4-1]));

        //c1[N/4-1][N/4] = guard0from1[N/2-1], c0[N/2][N/2-1]=guard0from2[N/2-1]
        y = -L/2.0 + (3*N/4-1)*dx;
        u = 1.414 * y;
        c14_new[N/4-1][N/4-1]=.25*(c14[N/4 - 2][N/4-1] + guard14from2[N/4-1] + c14[N/4-1][N/4-2] + guard14from15[N/4-1])
        -dt/(2*dx) * (u * (guard14from2[N/4-1]-c14[N/4 - 2][N/4-1]) + v * (guard14from15[N/4-1]-c14[N/4-1][N/4-2]));

    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/4;i++)
            {
            for(int j=0;j<N/4;j++)
                fprintf(data14,"%.4f ",c14_new[i][j]); 
            fprintf(data14,"\n"); 
            }
       } 

    pt14=c14;
    c14=c14_new;
    c14_new=pt14;
    }
t30 = omp_get_wtime();
fclose(data14);
}


else if(mype==15)
{
FILE *data15=fopen("lax15.txt", "w");
if(data15==NULL)
    {
    printf("Error opening file!\n");
    return 1;  // Indicate failure
    }

//write initial data to file
for(int i = 0; i<N/4; i++)
    { 
    for(int j=0; j<N/4; j++)  
        fprintf(data15,"%.4f ",c15[i][j]);
    fprintf(data15,"\n");
    }

t31 = omp_get_wtime();
for(int n=2;n <= NT;n++)
    {
    //prepare data to send to process #1 and #2
    for(int j=0;j<N/4;j++)
        {
        send15to3[j]=c15[N/4-1][j];
        send15to11[j]=c15[0][j];
        send15to12[j]=c15[j][N/4-1];
        send15to14[j]=c15[j][0];
        }

    //MPI COMM with #3
    MPI_Send(send15to3,N/4,MPI_DOUBLE,3,n,MPI_COMM_WORLD);
    MPI_Recv(guard15from3, N/4,MPI_DOUBLE,3,n,MPI_COMM_WORLD,&status);

    //MPI COMM with #11
    MPI_Send(send15to11,N/4,MPI_DOUBLE,11,n,MPI_COMM_WORLD);
    MPI_Recv(guard15from11, N/4,MPI_DOUBLE,11,n,MPI_COMM_WORLD,&status);
 

    //MPI COMM with #12
    MPI_Send(send15to12,N/4,MPI_DOUBLE,12,n,MPI_COMM_WORLD);
    MPI_Recv(guard15from12, N/4,MPI_DOUBLE,12,n,MPI_COMM_WORLD,&status);

 
    //MPI COMM with #14
    MPI_Send(send15to14, N/4,MPI_DOUBLE,14,n,MPI_COMM_WORLD);
    MPI_Recv(guard15from14, N/4,MPI_DOUBLE,14,n,MPI_COMM_WORLD,&status);



    //compute c15_new[0][j].for j=1,2,..,N/2-2, c2[N/2-1][*]
    //c0[-1][j] = guard0from2[N/2 + j]
    x= -L/2 + (3*N/4) * dx;
    v=(-1.414)*x;
    for(int j=1;j<N/4-1;j++)
        {
        y = -L/2.0 + (3*N/4 + j)*dx;
        u = 1.414 * y;
        c15_new[0][j]=.25*(guard15from11[j] + c15[1][j] + c15[0][j-1] + c15[0][j+1]) 
        - dt/(2*dx) * (u * (c15[1][j]-guard15from11[j]) + v * (c15[0][j+1]-c15[0][j-1]));
        }

    //compute c15_new[0][0] , need to use c[-1][0] = guard0from2[N/2] and c[0][-1]=guard0from1[N/2]
    y = -L/2.0 + (3*N/4)*dx;
    u = 1.414 * y;
    c15_new[0][0]=.25*(guard15from11[0] + c15[1][0] + guard15from14[0] + c15[0][1])
    - dt/(2*dx) * (u* (c15[1][0]-guard15from11[0]) + v* (c15[0][1]-guard15from14[0]));


    //compute c15_new[0][N/4-1], need to use c[0][N/2]=guard0from1[0], c[-1][N/2-1]=guard0from2[N-1]
    y = -L/2.0 + dx * (N-1);
    u = 1.414 * y;
    c15_new[0][N/4-1]=.25*(guard15from11[N/4-1] + c15[1][N/4-1] + c15[0][N/4-2] + guard15from12[0])
    - dt/(2*dx) * ((u*(c15[1][N/4-1] - guard15from11[N/4-1])) + v* (guard15from12[0] - c15[0][N/4-2]));


    #pragma omp parallel for default(shared) schedule(static) num_threads(nt)
    //compute c15_new from row 1 to row N/4-2
    for(int i=1;i<N/4-1;i++)
       { 
        x= -L/2.0 + (3*N/4+i)*dx;
        v= -1.414 * x;

        //compute c3_new[i][0]
        y= -L/2 + (3*N/4)*dx; 
        u= 1.414*y;
        c15_new[i][0]=.25*(c15[i-1][0] + c15[i+1][0] + guard15from14[i] + c15[i][1])
        - dt/(2*dx) * (u * (c15[i+1][0]-c15[i-1][0]) + v * (c15[i][1]-guard15from14[i]));
        //compute c3_new[i][j] for j=1,2..,N/4-2.
        for(int j=1;j<N/4-1;j++)
            {
            y = -L/2.0 + (3*N/4+j)*dx;
            u = 1.414 * y;
            c15_new[i][j]=.25*(c15[i-1][j] + c15[i+1][j] + c15[i][j-1] + c15[i][j+1]) 
            - dt/(2*dx) * (u * (c15[i+1][j]-c15[i-1][j]) + v * (c15[i][j+1]-c15[i][j-1]));
            }

        //compute c3_new[i][N/4-1]
        y=-L/2 + (N-1)*dx;
        u = 1.414*y;
        c15_new[i][N/4-1]=.25*(c15[i-1][N/4-1] + c15[i+1][N/4-1] + c15[i][N/4-2] + guard15from12[i]) 
        - dt/(2*dx) * (u * (c15[i+1][N/4-1]-c15[i-1][N/4-1]) + v * (guard15from12[i]-c15[i][N/4-2]));

        }

        //compute c15_new[N/4-1][*]. Need to use c[N/2][j] = guard0from2[j] for j = 1,2..N/2-2
        x = -L/2 + (N - 1)*dx;
        v = -1.414 * x;
        for(int j=1; j<N/4-1; j++)
            {
            y = -L/2.0 + (3*N/4 + j)*dx;
            u = 1.414 * y;
            c15_new[N/4-1][j]=.25*(c15[N/4 - 2][j] + guard15from3[j] + c15[N/4 - 1][j-1] + c15[N/4 - 1][j+1])
            -dt/(2*dx) * (u * (guard15from3[j]-c15[N/4 - 2][j]) + v * (c15[N/4 - 1][j+1]-c15[N/4 - 1][j-1]));
            }
        //compute c15_new[N/4-1][0], c0[N/2][0]=guard0from2[0], c0[N/2-1][-1]=guard0from1[N-1]
        y = -L/2.0 + (3*N/4)*dx;
        u = 1.414 * y;
        c15_new[N/4-1][0]=.25*(c15[N/4 - 2][0] + guard15from3[0] + guard15from14[N/4-1] + c15[N/4 - 1][1])
        -dt/(2*dx) * (u * (guard15from3[0]-c15[N/4 - 2][0]) + v * (c15[N/4 - 1][1]-guard15from14[N/4-1]));

        //c1[N/4-1][N/4] = guard0from1[N/2-1], c0[N/2][N/2-1]=guard0from2[N/2-1]
        y = -L/2.0 + (N-1)*dx;
        u = 1.414 * y;
        c15_new[N/4-1][N/4-1]=.25*(c15[N/4 - 2][N/4-1] + guard15from3[N/4-1] + c15[N/4-1][N/4-2] + guard15from12[N/4-1])
        -dt/(2*dx) * (u * (guard15from3[N/4-1]-c15[N/4 - 2][N/4-1]) + v * (guard15from12[N/4-1]-c15[N/4-1][N/4-2]));

    if(n==NT/2||n==NT)
       {
        for(int i=0;i<N/4;i++)
            {
            for(int j=0;j<N/4;j++)
                fprintf(data15,"%.4f ",c15_new[i][j]); 
            fprintf(data15,"\n"); 
            }
       } 

    pt15=c15;
    c15=c15_new;
    c15_new=pt15;
    }
t32 = omp_get_wtime();
fclose(data15);
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


return 0;
}
