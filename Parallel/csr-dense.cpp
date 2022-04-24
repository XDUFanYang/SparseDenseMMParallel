#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "matrix.h"
#include "coo.h"
#include "csr.h"

using namespace std;

#define N 5

int *a, *b, *c;
sparse_csr csr;
// csr multiply dense
// b : dst dense matrix
// a : src dense matrix
void CsrDenseMatrixMultiply(int m, int n, int k, int *A, int lda, int *B, int ldb, const sparse_csr & csr)
{
    //遍历每一个行指针 就是从第一行开始的 
    for(int i=0;i<csr.rpl-1;i++){
        for(int j=csr.row_ptr[i];j<csr.row_ptr[i+1];j++){
            //现在是第i行 
            //第col_index[j]列
            int csr_x=i;
            int csr_y=csr.col_index[j];
            int csr_v=csr.val[j];
            for(int k=0;k<n;k++){
                B(csr_x,k)+=csr_v*A(csr_y,k);
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

    
    int r1[6]={0,1,2,3,4,5};
    int c1[5]={0,1,2,3,4};
    int v1[5]={1,2,3,4,5};

    //void initcsr(sparse_csr & csr, int l, int *rp, int *ci, int *v, int rpl)
    initcsr(csr,N,r1,c1,v1,N+1);
    printcsr(csr);
    CsrDenseMatrixMultiply(N,N,N,a,N,b,N,csr);
    printf("--------res:------\n");
    print_matrix(N,N,b,N);

    return 0;
}