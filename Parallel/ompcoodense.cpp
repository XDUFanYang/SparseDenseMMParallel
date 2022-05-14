#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <sys/time.h>

#include "matrix.h"
#include "coo.h"
#include "csr.h"


using namespace std;

#define N 20000

float *a, *b, *c;

int thread_count;

int lda=N,ldb=N,ldc=N;

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

    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0) {
            printf("Using %i thread(s)\n", omp_get_num_threads());
        }
        #pragma omp parallel for
    for(int i=0;i<csr.rpl-1;i++){
        // int tid = omp_get_thread_num();
        // std::cout<<tid<<"\t tid"<<std::endl;
        for(int j=csr.row_ptr[i];j<csr.row_ptr[i+1];j++){
            int csr_x=i;
            int csr_y=csr.col_index[j];
            int csr_v=csr.val[j];
            for(int k=0;k<n;k++){
                B(csr_x,k)+=csr_v*A(csr_y,k);
            }
        }
        // std::cout<<tid<<"\t task finished"<<std::endl;
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

void OMPCooDenseMatrixMultiply(int m, int n, int k, float *A, int lda, float *B, int ldb, const sparse_coo & sc)
{

    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0) {
            printf("Using %i thread(s)\n", omp_get_num_threads());
        }

        #pragma omp for
        for(int coo_l=0;coo_l<sc.len;coo_l++){
        // int tid = omp_get_thread_num();
        // std::cout<<tid<<"\t tid"<<std::endl;
        int coo_x=sc.row_index[coo_l];
        int coo_y=sc.col_index[coo_l];
        int coo_val=sc.val[coo_l];
        for(int i=0;i<n;i++){
            B(coo_x,i)+=coo_val*A(coo_y,i);
        }
        // std::cout<<tid<<"\t task finished"<<std::endl;
        }
    }
}

void* coodensesubmul(void* rank){    
	/* Compute Block Size */
	long k = (long) rank;

    int start = (int(coo.len/thread_count))*k;
    int end = (int(coo.len/thread_count))*(k+1);
    for(int coo_l=start;coo_l<end;coo_l++){
        int coo_x=coo.row_index[coo_l];
        int coo_y=coo.col_index[coo_l];
        int coo_val=coo.val[coo_l];
        for(int i=0;i<N;i++){
            B(coo_x,i)+=coo_val*A(coo_y,i);
        }
    }
    return 0;
}

void* csrdensesubmul(void* rank){    
	/* Compute Block Size */
	long k = (long) rank;
    int start = (int(csr.rpl/thread_count))*k;
    int end = (int(csr.rpl/thread_count))*(k+1);
    for(int i=start;i<end;i++){
        for(int j=csr.row_ptr[i];j<csr.row_ptr[i+1];j++){
            int csr_x=i;
            int csr_y=csr.col_index[j];
            int csr_v=csr.val[j];
            for(int k=0;k<N;k++){
                B(csr_x,k)+=csr_v*A(csr_y,k);
            }
        }
    }
    return 0;
}

int main(int argc, char* argv[])
{   
    /* Get number of threads from command line */
    thread_count = strtol(argv[1], NULL, 10);
    printf("the thread_count is %d\n",thread_count);

    int nProcessors = omp_get_max_threads();
    std::cout<<nProcessors<<std::endl;
    omp_set_num_threads(10);

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

    struct timeval start,end;  
    gettimeofday(&start, NULL );
    CooDenseMatrixMultiply(N,N,N,a,N,b,N,coo);
    gettimeofday(&end, NULL );
    double timeuse = ( end.tv_sec - start.tv_sec ) + (end.tv_usec - start.tv_usec)/1000000.0;  
    printf("CooDense MM timing:%fs\n",timeuse);

    random_matrix(N,N,a,N);
    ones_matrix(N,N,b,N);
    struct timeval start1,end1;  
    gettimeofday(&start1, NULL );
    OMPCooDenseMatrixMultiply(N,N,N,a,N,b,N,coo);
    gettimeofday(&end1, NULL );
    double timeuse1 = ( end1.tv_sec - start1.tv_sec ) + (end1.tv_usec - start1.tv_usec)/1000000.0;  
    printf("OpenMPCooDense MM timing:%fs\n",timeuse1);
    
    random_matrix(N,N,a,N);
    ones_matrix(N,N,b,N);
    long thread; /* Use long in case of a 64-bit system */
    pthread_t* thread_handles;
    thread_handles = (pthread_t*)malloc(thread_count*sizeof(pthread_t));
    struct timeval start2,end2;  
    gettimeofday(&start2, NULL );
    for (thread = 0; thread < thread_count; thread++){

        if(pthread_create(&thread_handles[thread], NULL, coodensesubmul, (void*) thread) !=0 ){
          perror("Can't create thread");
      		free(thread_handles);
      		exit(-1);
        }
    }
    // printf("Hello from the main thread\n");
    for (thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);
    free(thread_handles);
    gettimeofday(&end2, NULL );
    double timeuse2 = ( end2.tv_sec - start2.tv_sec ) + (end2.tv_usec - start2.tv_usec)/1000000.0;  
    printf("CooDensePthread MM timing:%fs\n",timeuse2);


    random_matrix(N,N,a,N);
    ones_matrix(N,N,b,N);
    Coo2Csr(coo,csr);
    struct timeval start3,end3;  
    gettimeofday(&start3, NULL );
    CsrDenseMatrixMultiply(N,N,N,a,N,b,N,csr);
    gettimeofday(&end3, NULL );
    double timeuse3 = ( end3.tv_sec - start3.tv_sec ) + (end3.tv_usec - start3.tv_usec)/1000000.0;  
    printf("CsrDense MM timing:%fs\n",timeuse3);

    random_matrix(N,N,a,N);
    ones_matrix(N,N,b,N);
    struct timeval start4,end4;  
    gettimeofday(&start4, NULL );
    OMPCsrDenseMatrixMultiply(N,N,N,a,N,b,N,csr);
    gettimeofday(&end4, NULL );
    double timeuse4 = ( end4.tv_sec - start4.tv_sec ) + (end4.tv_usec - start4.tv_usec)/1000000.0;  
    printf("OMPCsrDense MM timing:%fs\n",timeuse4);
    
    
    random_matrix(N,N,a,N);
    ones_matrix(N,N,b,N);
    Coo2Csr(coo,csr);
    struct timeval start5,end5;  
    gettimeofday(&start5, NULL );
    thread_handles = (pthread_t*)malloc(thread_count*sizeof(pthread_t));
    for (thread = 0; thread < thread_count; thread++){

        if(pthread_create(&thread_handles[thread], NULL, csrdensesubmul, (void*) thread) !=0 ){
          perror("Can't create thread");
      		free(thread_handles);
      		exit(-1);
        }
    }
    // printf("Hello from the main thread\n");
    for (thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);
    free(thread_handles);
    gettimeofday(&end5, NULL );
    double timeuse5 = ( end5.tv_sec - start5.tv_sec ) + (end5.tv_usec - start5.tv_usec)/1000000.0;  
    printf("CsrDense pthread timing:%fs\n",timeuse5);
    return 0;
}