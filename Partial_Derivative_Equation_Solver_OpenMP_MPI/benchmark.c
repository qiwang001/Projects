#include "stdio.h"
#include "stdlib.h"
#include "assert.h"
//#include "mpi.h"
#include"omp.h"
#include"math.h"

double**
datamatrix(unsigned long n)
{
    double *data = (double*)calloc(n * n, sizeof(double));
    double **M = (double**)malloc(n * sizeof(double*)); 
    for (unsigned long i=0; i < n; i++)
        M[i] = data + n * i;
        //M[i] = &data[n * i];
    return M;
}

double
lax(double**c, double L,int N, int NT,double u,double v,double dx,double dt,int nt)
{
    double x,y;
    double**c1 = datamatrix(N);
    double**t;
    int im1,ia1,jm1,ja1;
    //#pragma omp parallel for default(none) shared(N,L,c,dx,x,y) schedule(static) num_threads(nt)
    FILE *data=fopen("lax.txt", "w");
    if(data==NULL)
        {
        printf("Error opening file!\n");
        return 1;  // Indicate failure
        }
    for(int i = 2*N/5; i<3*N/5; i++)
        {
        for(int j = 0;j < N; j++)
            c[i][j] = 1;
        }
    /*
    for(int i = 0; i<N; i++)
        { 
        for(int j=0; j<N; j++)  
            fprintf(data,"%.4f ",c[i][j]);
        fprintf(data,"\n");
        }*/
    //}
    //#pragma omp parallel for default(none)  shared(data,c,c1,N,NT,dx,dt,u,v,im1,ia1,jm1,ja1,x,y) schedule(static) num_threads(nt)
    
    
    double t1=omp_get_wtime();
    //#pragma omp parallel for default(none) shared(c,c1) private(x,y,L,im1,ia1,jm1,ja1,N,NT,dx,dt,u,v,t) schedule(static) num_threads(nt)
    #pragma omp parallel for default(none) shared(t1,data,c,c1,L,N,NT) private(im1,ia1,jm1,ja1,dx,dt,u,y,t,x,v) schedule(static) num_threads(nt)
    for(int n=2;n <= NT;n++)
      {  
        if(n%100==0)
            {
                printf("going well on %d \n",n);
                printf("time passed: %fs\n", omp_get_wtime()-t1);
            }
        for(int i=0;i<N; i++)
            {
                im1=(i-1)%N;
                if (im1<0)
                    im1+=N;
                ia1=(i+1)%N;
                x = -L/2.0 + i*dx;
                v = -1.414 * x;
                for(int j=0;j < N;j++)
                    {
                    y=-L/2.0 + j*dx;
                    u = 1.414*y;
                    jm1=(j-1)%N;
                    if (jm1<0)
                        jm1+=N;
                    ja1 = (j + 1)%N;
                    c1[i][j]=.25*(c[im1][j] + c[ia1][j] + c[i][jm1] + c[i][ja1]) + 
                    - dt/(2*dx) * (u * (c[ia1][j]-c[im1][j]) + v * (c[i][ja1]-c[i][jm1]));
                    //if(n==NT/2||n==NT)
                   //     fprintf(data,"%.4f ",c1[i][j]);
                    }
               // if(n==NT/2||n==NT)
                 //   fprintf(data,"\n ");
            }
      t=c1;
      c1=c;
      c=t;
      }
    double t2=omp_get_wtime();
  
    free(c1[0]);
    free(c1);
    return t2 - t1;
}
//int main()
int main(int argc, char* argv[])
{
    int N,nt,NT;
    unsigned long i,j;
    N=atoi(argv[1]);
    nt=atoi(argv[2]);
    double L=1, T=1, u=1, v=1, dx, dt;
    dx = L / (N-1);
    dt = 1.25e-4;
    NT = 1.0/ dt;
    double **c = datamatrix(N);
    double delt = lax(c, L, N, NT, u, v, dx, dt, nt);
    double Nd=N;
    double NTd=NT;
    printf("grind rate: %.2fM\n, time used: %.2f\n",Nd*Nd*NTd/(delt*1000000),delt);
    free(c[0]);
    free(c);
    return 0;
}