#ifndef _MATRIX_H
# define _MATRIX_H

#include <stdio.h>
#include <cstdlib>

#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]
#define C( i, j ) c[ (i)*ldc + (j) ]


// gemm C = A * B + C
void MatrixMultiply(int m, int n, int k, int *a, int lda, int *b, int ldb, int *c, int ldc)
{
    for(int i = 0; i < m; i++){
        for (int p=0; p<k; p++ ){ 
            for (int j=0; j<n; j++ ){    
                C(i, j) = C(i, j) + A(i, p) * B(p, j);
            }
        }
    }
}

void random_matrix( int m, int n, int *a, int lda )
{
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
        A(i, j) = i;
    }
  }
}

void zero_matrix( int m, int n, int *a, int lda )
{
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
        A(i, j) = 0;
    }
  }
}

void print_matrix( int m, int n, int *a, int lda ){
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            printf("%d ",A(i,j));
        }
        printf("\n");
    }
}

#endif