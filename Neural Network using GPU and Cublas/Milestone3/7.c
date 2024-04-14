#include <stdio.h>

#include <stdlib.h>
#include"/opt/homebrew/Cellar/openblas/0.3.26/include/cblas.h"


int main() {
  // Define matrix dimensions (assuming float32 for simplicity)
  int m = 3;  // Rows in matrix A and resulting C
  int n = 2;  // Columns in matrix B and resulting C
  int k = 4;  // Columns in matrix A and rows in matrix B

  // Allocate memory for matrices (row-major order)
  float* A = (float*)malloc(m * k * sizeof(float));
  float* B = (float*)malloc(k * n * sizeof(float));
  float* C = (float*)malloc(m * n * sizeof(float));

  // Initialize matrices with some sample values (replace with your actual data)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      A[i * k + j] = i + j * 0.1f;  // Sample values for A
    }
  }

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      B[i * n + j] = i * 0.2f + j;  // Sample values for B
    }
  }

  // Set values for alpha and beta (scalars used in the multiplication)
  float alpha = 1.0f;
  float beta = 0.0f;  // No contribution from previous matrix

  // Perform matrix multiplication using cblas_sgemm (C = alpha * A * B + beta * C)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,m, n, k, alpha, A, k, B, n, beta, C, n);

  // Print the resulting matrix C
  printf("Resultant matrix C:\n");
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f ", C[i * n + j]);
    }
    printf("\n");
  }

  // Free allocated memory
  free(A);
  free(B);
  free(C);

  return 0;
}
