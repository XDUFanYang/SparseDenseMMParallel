/* POLYBENCH/GPU-OPENMP
 *
 * This file is a part of the Polybench/GPU-OpenMP suite
 *
 * Contact:
 * William Killian <killian@udel.edu>
 * 
 * Copyright 2013, The University of Delaware
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>


#include "matrix.h"

#define N 10

using namespace std;

int *a, *b, *c;
int main()
{
    a = (int *)malloc(N * N * sizeof(int));
    b = (int *)malloc(N * N * sizeof(int));
    c = (int *)malloc(N * N * sizeof(int));
    random_matrix(N,N,a,N);
    random_matrix(N,N,b,N);
    random_matrix(N,N,c,N);
    
    printf("--------a------\n");
    print_matrix(N,N,a,N);
    
    printf("--------b------\n");
    print_matrix(N,N,b,N);

    printf("--------c------\n");
    print_matrix(N,N,c,N);

    printf("--------C=C+AB------\n");
    MatrixMultiply(N,N,N,a,N,b,N,c,N);
    print_matrix(N,N,c,N);
    return 0;
}
