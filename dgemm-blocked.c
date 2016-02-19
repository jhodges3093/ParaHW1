/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = icc 
OPT = -O3 -funroll-loops -ftree-vectorize-verbose
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/apps/intel/15/composer_xe_2015.2.164/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/

#include<cmath.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE floor(sqrt(lda))
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; i+4)
  {
    /* For each column j of B */ 
    for (int j = 0; j < N; j+4) 
    {
      /* Compute C(i,j) */
      double cij = C[i+j*lda];
	  double cijTwo = C[(i+1)+(j+1)*lda];
	  double cijThree = C[(i+2)+(j+2)*lda];
	  double cijFour = C[(i+3)+(j+3)*lda];
      for (int k = 0; k < K; k+4)
		cij += A[i+k*lda] * B[k+j*lda];
		cijTwo += A[(i+1)+[k+1]*lda] * B[(k+1)+(j+1)*lda];
		cijThree += A[(i+2)+[k+2]*lda] * B[(k+2)+(j+2)*lda];
		cijFour += A[(i+3)+[k+3]*lda] * B[(k+3)+(j+3)*lda];
      C[i+j*lda] = cij;
	  C[(i+1)+(j+1)*lda] = cijTwo;
	  C[(i+2)+(j+2)*lda] = cijThree;
	  C[(i+3)+(j+3)*lda] = cijFour;
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE, lda-i);
	int N = min (BLOCK_SIZE, lda-j);
	int K = min (BLOCK_SIZE, lda-k);

	/* Perform individual block dgemm */
	do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}
