#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "matrix.h"
#include "coo.h"


using namespace std;

#define N 5

int *a, *b, *c;
sparse_coo coo;

// coo multiply dense
// b : dst dense matrix
// a : src dense matrix
void CooDenseMatrixMultiply(int m, int n, int k, int *A, int lda, int *B, int ldb, const sparse_coo & sc)
{
    for(int coo_l=0;coo_l<sc.len;coo_l++){
        int coo_x=sc.row_index[coo_l];
        int coo_y=sc.col_index[coo_l];
        int coo_val=sc.val[coo_l];
        for(int i=0;i<n;i++){
            B(coo_x,i)+=coo_val*A(coo_y,i);
        }
    }
}
int main()
{
    a= (int *)malloc(N * N * sizeof(int));
    b= (int *)malloc(N * N * sizeof(int));
    random_matrix(N,N,a,N);
    zero_matrix(N,N,b,N);
    printf("--------a------\n");
    print_matrix(N,N,a,N);
    printf("--------b------\n");
    print_matrix(N,N,b,N);

    init(coo,100);
    insert(0,1,1,coo);
    insert(0,2,3,coo);
    insert(1,3,10,coo);
    printcoo(coo);

    CooDenseMatrixMultiply(N,N,N,a,N,b,N,coo);
    printf("--------res:------\n");
    print_matrix(N,N,b,N);

    return 0;
}