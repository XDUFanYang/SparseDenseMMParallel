#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "matrix.h"
#include "coo.h"
#include "csr.h"
#include "csc.h"

using namespace std;

#define N 5

int *a, *b, *c;
sparse_csc csc;
// csc multiply dense
// b : dst dense matrix
// a : src dense matrix
void CscDenseMatrixMultiply(int m, int n, int k, int *A, int lda, int *B, int ldb, const sparse_csc & csc)
{
    for(int i=0;i<csc.cpl-1;i++){
        for(int j=csc.col_ptr[i];j<csc.col_ptr[i+1];j++){
            int csc_y=i;
            int csc_x=csc.row_index[j];
            int csc_v=csc.val[j];
             for(int k=0;k<n;k++){
                B(csc_x,k)+=csc_v*A(csc_y,k);
                // printf("( %d,%d ) update + %d\n",csr_x,csr_y,csr_v*A(csr_y,k));
            }
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

    
    int cp[6]={0,1,2,3,4,5};
    int r1[5]={0,1,2,3,4};
    int v1[5]={1,2,3,4,5};

    //void initcsr(sparse_csr & csr, int l, int *rp, int *ci, int *v, int rpl)
    initcsc(csc,N,cp,r1,v1,N+1);
    printcsc(csc);
    CscDenseMatrixMultiply(N,N,N,a,N,b,N,csc);
    printf("--------res:------\n");
    print_matrix(N,N,b,N);

    return 0;
}