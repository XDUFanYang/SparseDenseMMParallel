#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "matrix.h"
#include "coo.h"
#include "csr.h"
#include "csbcsr.h"

using namespace std;

#define N 4

int *a, *b, *c;
sparse_csbcsr bcsr;
// csr multiply dense
// b : dst dense matrix
// a : src dense matrix
void BcsrDenseMatrixMultiply(int m, int n, int k, int *A, int lda, int *B, int ldb, const sparse_csbcsr & bcsr)
{
    //遍历每个行列指针
    for(int i=0;i<bcsr.block_num;i++){
        int start_x=bcsr.rows[i];
        int start_y=bcsr.cols[i];
        for(int j=0;j<bcsr.length;j++){
            for(int k=0;k<bcsr.length;k++){
                int row=start_x+j;
                int col=start_y+k;
                int val=bcsr.val[(i*bcsr.length*bcsr.length+j*bcsr.length+k)];
                //子矩阵从(row,col)开始
                for(int t=0;t<n;t++){
                    B(row,t)+=val*A(col,t);
                }
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

    
    int rs[2]={0,2};
    int cs[2]={0,2};
    int v[8]={1,0,0,2,3,0,0,4};

    //void initcsr(sparse_csr & csr, int l, int *rp, int *ci, int *v, int rpl)
    initcsbcsr(bcsr,2,2,rs,cs,v);
    printcsbcsr(bcsr);
    BcsrDenseMatrixMultiply(N,N,N,a,N,b,N,bcsr);
    printf("--------res:------\n");
    print_matrix(N,N,b,N);

    return 0;
}