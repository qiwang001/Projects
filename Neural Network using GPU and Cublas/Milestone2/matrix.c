#define e 2.71828
#include "mnist.h"
#include "omp.h"
#include<time.h>
#include "math.h"
#include <stdio.h>
#include"/opt/homebrew/Cellar/openblas/0.3.26/include/cblas.h"
#include<math.h>

void blas_mm(double * A, double * B, double * C, int m, int k, int n, double alpha,double beta)
{
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, alpha, A, k, B, n, beta, C, n);
}

void hand_mm(double * A, double * B, double * C, int m, int k, int n)
{
  int cur_row=0, cur_col=0;
  for(int i=0;i < m * n;i++)
    {
     cur_row = i/n;
     cur_col = i - cur_row * n;
     for(int j=0;j<k;j++)
           C[i] += A[cur_row * k + j] * B[cur_col + j * n];
    } 
}

double* trans(double * M, int row, int col)
{
    double * MT = (double*) malloc(sizeof(double) * row *col);
    for(int i=0;i<row;i++)
      for(int j=0;j<col;j++)
        MT[j*row+i]= M[i*col+j];
    
    free(M);
    return MT;
}

int
main()
{
    double *A,*B,*C;
    int m= 3, k=4, n=5,i,j;
    A = (double *)malloc( m*k*sizeof( double ));
    B = (double *)malloc( k*n*sizeof( double ) );
    C = (double *)malloc( m*n*sizeof( double ) );
    for (i = 0; i < (m*k); i++) {
        A[i] = (double)(i+1);
    }

    for (i = 0; i < (k*n); i++) {
        B[i] = (double)(-i-1);
    }

    for (i = 0; i < (m*n); i++) {
        C[i] = 0.0;
    }

    hand_mm(A,B,C,m,k,n);

    printf ("\n Top left corner of matrix C: \n");
    for (i=0; i<3; i++) {
      for (j=0; j<5; j++) {
        printf ("%1.3f ", C[j+i*n]);
      }
      printf ("\n");
    }


    for (i = 0; i < (m*k); i++) {
        A[i] = (double)(i+1);
    }

    for (i = 0; i < (k*n); i++) {
        B[i] = (double)(-i-1);
    }

    for (i = 0; i < (m*n); i++) {
        C[i] = 0.0;
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, 1, A, k, B, n, 1, C, n);

    printf ("\n Top left corner of matrix C: \n");
    for (i=0; i<3; i++) {
      for (j=0; j<5; j++) {
        printf ("%1.3f ", C[j+i*n]);
      }
      printf ("\n");
    }

    double*CT = trans(C,3,5);

    printf ("\n After transpose: \n");
    for (i=0; i<5; i++) {
      for (j=0; j<3; j++) {
        printf ("%1.3f ", CT[j+i*3]);
      }
      printf ("\n");
    }



}