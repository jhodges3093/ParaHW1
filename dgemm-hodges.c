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
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

const char* dgemm_desc = "Naive, Matrix Multiply.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_dgemm (int n, double* A, double* B, double* C)
{
  
  MatrixXd AX;
  MatrixXd BX;
  MatrixXd CX;
  
  MatrixXd AX = Map<MatrixXd>( A, n, n );
  MatrixXd BX = Map<MatrixXd>( B, n, n );
  MatrixXd CX = Map<MatrixXd>( B, n, n );
  
  CX += AX * BX;
  
  Map<MatrixXd>( C, CX.rows(), CX.cols() ) = CX;
    
}
