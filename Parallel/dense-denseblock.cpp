#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <xmmintrin.h>
#include <arm_neon.h>

#include "matrix.h"

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

#define N 1000

float *a, *b, *c;

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

void SimdBlock( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc ){
  
  float 
    /* Point to the current elements in the four rows of A */
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

int main(){
    a = (float *)malloc(N * N * sizeof(float));
    b = (float *)malloc(N * N * sizeof(float));
    c = (float *)malloc(N * N * sizeof(float));
    random_matrix(N,N,a,N);
    random_matrix(N,N,b,N);
    zero_matrix(N,N,c,N);

    clock_t start,finish;
    start=clock();
    MatrixMultiply(N,N,N,a,N,b,N,c,N);
    finish=clock();
    printf("traditional MM timing:%fs\n",(double)(finish - start) / CLOCKS_PER_SEC);

    random_matrix(N,N,a,N);
    random_matrix(N,N,b,N);
    zero_matrix(N,N,c,N);
    
    clock_t start1,finish1;
    start1=clock();
    for(int i=0;i<N;i+=4){
      for(int j=0;j<N;j+=4){
        int lda=N,ldb=N,ldc=N;
        Block44MatrixMultiply(N, &A(i,0), N, &B(0,j), N, &C(i,j), N);
      }
    }
    finish1=clock();
    printf("block 4*4 MM timing:%fs\n",(double)(finish1 - start1) / CLOCKS_PER_SEC);
    
    random_matrix(N,N,a,N);
    random_matrix(N,N,b,N);
    zero_matrix(N,N,c,N);
    clock_t start2,finish2;
    start2=clock();
    for(int i=0;i<N;i+=4){
      for(int j=0;j<N;j+=4){
        int lda=N,ldb=N,ldc=N;
        Block44OptimizeReg(N, &A(i,0), N, &B(0,j), N, &C(i,j), N);
      }
    }
    finish2=clock();
    printf("blockOptimizeReg 4*4 MM timing:%fs\n",(double)(finish2 - start2) / CLOCKS_PER_SEC);


    random_matrix(N,N,a,N);
    random_matrix(N,N,b,N);
    zero_matrix(N,N,c,N);

    clock_t start3,finish3;
    start3=clock();
    for(int i=0;i<N;i+=4){
      for(int j=0;j<N;j+=4){
        int lda=N,ldb=N,ldc=N;
        SimdBlock(N, &A(i,0), N, &B(0,j), N, &C(i,j), N);
      }
    }
    finish3=clock();
    printf("SimdOptimize timing:%fs\n",(double)(finish3 - start3) / CLOCKS_PER_SEC);
    return 0;
}
