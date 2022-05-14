#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <xmmintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <omp.h>
#include <sys/time.h>
#include <arm_neon.h>

#include "matrix.h"

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

#define N 500

float *a, *b, *c;

int thread_count;

int lda=N, ldb=N,ldc=N; 

void Block44MatrixMultiply(int k, float *a, int lda,  float *b, int ldb, float *c, int ldc){
    int p;
    for ( p=0; p<k; p++ ){
      C( 0, 0 ) += A( 0, p ) * B( p, 0 );     
      C( 0, 1 ) += A( 0, p ) * B( p, 1 );     
      C( 0, 2 ) += A( 0, p ) * B( p, 2 );     
      C( 0, 3 ) += A( 0, p ) * B( p, 3 );     

      C( 1, 0 ) += A( 1, p ) * B( p, 0 );     
      C( 1, 1 ) += A( 1, p ) * B( p, 1 );     
      C( 1, 2 ) += A( 1, p ) * B( p, 2 );     
      C( 1, 3 ) += A( 1, p ) * B( p, 3 );     

      C( 2, 0 ) += A( 2, p ) * B( p, 0 );     
      C( 2, 1 ) += A( 2, p ) * B( p, 1 );     
      C( 2, 2 ) += A( 2, p ) * B( p, 2 );     
      C( 2, 3 ) += A( 2, p ) * B( p, 3 );     

      C( 3, 0 ) += A( 3, p ) * B( p, 0 );     
      C( 3, 1 ) += A( 3, p ) * B( p, 1 );     
      C( 3, 2 ) += A( 3, p ) * B( p, 2 );     
      C( 3, 3 ) += A( 3, p ) * B( p, 3 );
    } 
}

void Block44OptimizeReg( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
  int p;
  register float
       c_00_reg,   c_01_reg,   c_02_reg,   c_03_reg,  
       c_10_reg,   c_11_reg,   c_12_reg,   c_13_reg,  
       c_20_reg,   c_21_reg,   c_22_reg,   c_23_reg,  
       c_30_reg,   c_31_reg,   c_32_reg,   c_33_reg,
       a_0p_reg,   a_1p_reg,   a_2p_reg,   a_3p_reg;

  c_00_reg = 0.0;   c_01_reg = 0.0;   c_02_reg = 0.0;   c_03_reg = 0.0;
  c_10_reg = 0.0;   c_11_reg = 0.0;   c_12_reg = 0.0;   c_13_reg = 0.0;
  c_20_reg = 0.0;   c_21_reg = 0.0;   c_22_reg = 0.0;   c_23_reg = 0.0;
  c_30_reg = 0.0;   c_31_reg = 0.0;   c_32_reg = 0.0;   c_33_reg = 0.0;

  for ( p=0; p<k; p++ ){
    a_0p_reg = A( 0, p ); a_1p_reg = A( 1, p ); a_2p_reg = A( 2, p );a_3p_reg = A( 3, p );

    c_00_reg += a_0p_reg * B( p, 0 );  c_01_reg += a_0p_reg * B( p, 1 );  c_02_reg += a_0p_reg * B( p, 2 );  c_03_reg += a_0p_reg * B( p, 3 );     

    c_10_reg += a_1p_reg * B( p, 0 );  c_11_reg += a_1p_reg * B( p, 1 );  c_12_reg += a_1p_reg * B( p, 2 );  c_13_reg += a_1p_reg * B( p, 3 );     

    c_20_reg += a_2p_reg * B( p, 0 );  c_21_reg += a_2p_reg * B( p, 1 );  c_22_reg += a_2p_reg * B( p, 2 );  c_23_reg += a_2p_reg * B( p, 3 );     

    c_30_reg += a_3p_reg * B( p, 0 );     
    c_31_reg += a_3p_reg * B( p, 1 );     
    c_32_reg += a_3p_reg * B( p, 2 );     
    c_33_reg += a_3p_reg * B( p, 3 );     
  }

  C( 0, 0 ) += c_00_reg;   C( 0, 1 ) += c_01_reg;   C( 0, 2 ) += c_02_reg;   C( 0, 3 ) += c_03_reg;
  C( 1, 0 ) += c_10_reg;   C( 1, 1 ) += c_11_reg;   C( 1, 2 ) += c_12_reg;   C( 1, 3 ) += c_13_reg;
  C( 2, 0 ) += c_20_reg;   C( 2, 1 ) += c_21_reg;   C( 2, 2 ) += c_22_reg;   C( 2, 3 ) += c_23_reg;
  C( 3, 0 ) += c_30_reg;   C( 3, 1 ) += c_31_reg;   C( 3, 2 ) += c_32_reg;   C( 3, 3 ) += c_33_reg;
}

void* matrixmul(void* rank){    
	long k = (long) rank;

  int start = (int(N/thread_count))*k;
  int end = (int(N/thread_count))*(k+1);
  for(int i=start;i<end;i++){
    float t = 0;
    for(int j=0;j<N;j++){
       for(int k=0;k<N;k++){
          t += A(i,k)*B(k,j); 
       }
       C(i,j)+=t;  
    }
  }
	return 0;
}

void do_multiply(){

    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0) {
            printf("Using %i thread(s)\n", omp_get_num_threads());
        }

        #pragma omp for
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                float dot_product = 0;
                for (int k = 0; k < N; ++k) {
                    dot_product += A(i,k) * B(j,k);
                }
                C(i,j) = dot_product;
            }
        }
    }
}

void SimdBlock( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc ){
  
  float 
    *a_0p_pntr, *a_1p_pntr, *a_2p_pntr, *a_3p_pntr;

  a_0p_pntr = &A(0, 0);
  a_1p_pntr = &A(1, 0);
  a_2p_pntr = &A(2, 0);
  a_3p_pntr = &A(3, 0);

  float32x4_t c_p0_sum = {0};
  float32x4_t c_p1_sum = {0};
  float32x4_t c_p2_sum = {0};
  float32x4_t c_p3_sum = {0};

  register float
    a_0p_reg,
    a_1p_reg,   
    a_2p_reg,
    a_3p_reg;

  for (int p = 0; p < k; ++p) {
    float32x4_t b_reg = vld1q_f32(&B(p, 0));

    a_0p_reg = *a_0p_pntr++;
    a_1p_reg = *a_1p_pntr++;
    a_2p_reg = *a_2p_pntr++;
    a_3p_reg = *a_3p_pntr++;

    c_p0_sum = vmlaq_n_f32(c_p0_sum, b_reg, a_0p_reg);
    c_p1_sum = vmlaq_n_f32(c_p1_sum, b_reg, a_1p_reg);
    c_p2_sum = vmlaq_n_f32(c_p2_sum, b_reg, a_2p_reg);
    c_p3_sum = vmlaq_n_f32(c_p3_sum, b_reg, a_3p_reg);
  }

  float *c_pntr = 0;
  c_pntr = &C(0, 0);
  float32x4_t c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_p0_sum);
  vst1q_f32(c_pntr, c_reg);

  c_pntr = &C(1, 0);
  c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_p1_sum);
  vst1q_f32(c_pntr, c_reg);

  c_pntr = &C(2, 0);
  c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_p2_sum);
  vst1q_f32(c_pntr, c_reg);

  c_pntr = &C(3, 0);
  c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_p3_sum);
  vst1q_f32(c_pntr, c_reg);
}

int main(int argc, char* argv[]){
    thread_count = strtol(argv[1], NULL, 10);
    printf("the thread_count is %d\n",thread_count);

    omp_set_num_threads(10);

    a = (float *)malloc(N * N * sizeof(float));
    b = (float *)malloc(N * N * sizeof(float));
    c = (float *)malloc(N * N * sizeof(float));
    random_matrix(N,N,a,N);
    random_matrix(N,N,b,N);
    zero_matrix(N,N,c,N);

    struct timeval start,end;  
    gettimeofday(&start, NULL );  
    MatrixMultiply(N,N,N,a,N,b,N,c,N);
    gettimeofday(&end, NULL ); 
    double timeuse = ( end.tv_sec - start.tv_sec ) + (end.tv_usec - start.tv_usec)/1000000.0;  
    printf("traditional MM time=%f\n",timeuse); 

    random_matrix(N,N,a,N);
    random_matrix(N,N,b,N);
    zero_matrix(N,N,c,N);
    
    struct timeval start1,end1;  
    gettimeofday(&start1, NULL ); 
    for(int i=0;i<N;i+=4){
      for(int j=0;j<N;j+=4){
        int lda=N,ldb=N,ldc=N;
        Block44MatrixMultiply(N, &A(i,0), N, &B(0,j), N, &C(i,j), N);
      }
    }
    gettimeofday(&end1, NULL); 
    double timeuse1 = ( end1.tv_sec - start1.tv_sec ) + (end1.tv_usec - start1.tv_usec)/1000000.0;  
    printf("block 4*4 MM time=%f\n",timeuse1); 
    
    random_matrix(N,N,a,N);
    random_matrix(N,N,b,N);
    zero_matrix(N,N,c,N);
    struct timeval start2,end2;  
    gettimeofday(&start2, NULL );
    for(int i=0;i<N;i+=4){
      for(int j=0;j<N;j+=4){
        int lda=N,ldb=N,ldc=N;
        Block44OptimizeReg(N, &A(i,0), N, &B(0,j), N, &C(i,j), N);
      }
    }
    gettimeofday(&end2, NULL); 
    double timeuse2 = ( end2.tv_sec - start2.tv_sec ) + (end2.tv_usec - start2.tv_usec)/1000000.0;  
    printf("blockOptimizeReg 4*4 MM time=%f\n",timeuse1); 


    random_matrix(N,N,a,N);
    random_matrix(N,N,b,N);
    zero_matrix(N,N,c,N);

    long thread; 
    pthread_t* thread_handles;
    thread_handles = (pthread_t*)malloc(thread_count*sizeof(pthread_t));
    struct timeval start3,end3;  
    gettimeofday(&start3, NULL );
    for (thread = 0; thread < thread_count; thread++){

        if(pthread_create(&thread_handles[thread], NULL, matrixmul, (void*) thread) !=0 ){
          perror("Can't create thread");
      		free(thread_handles);
      		exit(-1);
        }
    }
    
    for (thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);
    free(thread_handles);
   
    gettimeofday(&end3, NULL );
    double timeuse3 = ( end3.tv_sec - start3.tv_sec ) + (end3.tv_usec - start3.tv_usec)/1000000.0;  
    printf("pthreadspeedup time=%f\n",timeuse3); 
    
    random_matrix(N,N,a,N);
    random_matrix(N,N,b,N);
    zero_matrix(N,N,c,N);
    struct timeval start4,end4;  
    gettimeofday(&start4, NULL );
    do_multiply();
    gettimeofday(&end4, NULL );
    double timeuse4 = ( end4.tv_sec - start4.tv_sec ) + (end4.tv_usec - start4.tv_usec)/1000000.0;  
    printf("pthreadspeedup time=%f\n",timeuse4); 


    struct timeval start5,end5;  
    gettimeofday(&start5, NULL );

      if (omp_get_thread_num() == 0) {
          printf("Using %i thread(s)\n", omp_get_num_threads());
      }
      #pragma omp parallel
      {
      #pragma omp for
      for(int i=0;i<N;i+=4){
        for(int j=0;j<N;j+=4){
          int lda=N,ldb=N,ldc=N;
          SimdBlock(N, &A(i,0), N, &B(0,j), N, &C(i,j), N);
        }
      }
      }
    gettimeofday(&end5, NULL );
    double timeuse5 = ( end5.tv_sec - start5.tv_sec ) + (end5.tv_usec - start5.tv_usec)/1000000.0;  
    printf("SimdOMPOptimize timing:%fs\n",timeuse5);
    return 0;
}