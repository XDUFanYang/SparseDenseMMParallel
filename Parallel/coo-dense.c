#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>


#include "matrix.h"
#include "coo.h"
#include "csr.h"


using namespace std;

#define N 15000

float *a, *b, *c;

sparse_coo coo;
sparse_csr csr;

void CsrDenseMatrixMultiply(int m, int n, int k, float *A, int lda, float *B, int ldb, const sparse_csr & csr)
{
    for(int i=0;i<csr.rpl-1;i++){
        for(int j=csr.row_ptr[i];j<csr.row_ptr[i+1];j++){
            int csr_x=i;
            int csr_y=csr.col_index[j];
            int csr_v=csr.val[j];
            for(int k=0;k<n;k++){
                B(csr_x,k)+=csr_v*A(csr_y,k);
            }
        }
    }
}

void Coo2Csr(const sparse_coo& coo, sparse_csr& csr){
    int *rp;
    rp=(int *)malloc((coo.len+1) * sizeof(int));
    int count=0;

    for(int i=0;i<coo.len;i++){
        rp[coo.row_index[i]+1]++;
        if(rp[coo.row_index[i]+1]==1){
            count++;
        }
    }
    for(int i=0;i<count;i++){
        rp[i+1]+=rp[i];
    }
    initcsr(csr, coo.len, rp, coo.col_index, coo.val, count+1);
}

void OMPCsrDenseMatrixMultiply(int m, int n, int k, float *A, int lda, float *B, int ldb, const sparse_csr & csr)
{
    #pragma omp parallel num_threads(MAX)
    for(int i=0;i<csr.rpl-1;i++){
        for(int j=csr.row_ptr[i];j<csr.row_ptr[i+1];j++){
            int csr_x=i;
            int csr_y=csr.col_index[j];
            int csr_v=csr.val[j];
            for(int k=0;k<n;k++){
                B(csr_x,k)+=csr_v*A(csr_y,k);
            }
        }
    }
}

void CooDenseMatrixMultiply(int m, int n, int k, float *A, int lda, float *B, int ldb, const sparse_coo & sc)
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

void MPCooDenseMatrixMultiply(int m, int n, int k, float *A, int lda, float *B, int ldb, const sparse_coo & sc)
{
    #pragma omp parallel num_threads(MAX)
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
    a= (float *)malloc(N * N * sizeof(float));
    b= (float *)malloc(N * N * sizeof(float));
    
    random_matrix(N,N,a,N);
    ones_matrix(N,N,b,N);

    int val_num=int(N*N*0.0018);
    init(coo,val_num);
    for(int i=0;i<val_num;i++){
        srand(i);
        int m_x=((int)rand())%N;
        int m_y=((int)rand())%N;
        insert(m_x,m_y,1,coo);
    }

    clock_t start,finish;
    start=clock();
    CooDenseMatrixMultiply(N,N,N,a,N,b,N,coo);
    finish=clock();
    printf("CooDense MM timing:%fs\n",(double)(finish - start) / CLOCKS_PER_SEC);

    random_matrix(N,N,a,N);
    ones_matrix(N,N,b,N);
    clock_t start1,finish1;
    start1=clock();
    MPCooDenseMatrixMultiply(N,N,N,a,N,b,N,coo);
    finish1=clock();
    printf("OpenMPCooDense MM timing:%fs\n",(double)(finish1 - start1) / CLOCKS_PER_SEC);
    
    random_matrix(N,N,a,N);
    ones_matrix(N,N,b,N);
    Coo2Csr(coo,csr);
    clock_t start2,finish2;
    start2=clock();
    CsrDenseMatrixMultiply(N,N,N,a,N,b,N,csr);
    finish2=clock();
    printf("CsrDense MM timing:%fs\n",(double)(finish2 - start2) / CLOCKS_PER_SEC);

    random_matrix(N,N,a,N);
    ones_matrix(N,N,b,N);
    clock_t start3,finish3;
    start3=clock();
    OMPCsrDenseMatrixMultiply(N,N,N,a,N,b,N,csr);
    finish3=clock();
    printf("OMPCsrDense MM timing:%fs\n",(double)(finish3 - start3) / CLOCKS_PER_SEC);
    return 0;
}
