#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>
#include <tgmath.h>
#include <omp.h>
#include <cblas.h>

// Routines:
//
// - ASum(): Calculates the absolute sum of a vector. Equal to the 1-norm
//
// - SubVec(): Substracts two vectors and saves it in vec1
//
// - CopyVec(): Copies one vector from vec to target
//
// - ScaleVec(): Multiplies a vector with a scalar
//
// - DotProduct(): Calculates the dotproduct between two vectors
//
// - MatrixVectorMult_X(): Calculates the product between a mxn matrix and
//                         a m sized vector. Saves the result in out. X may
//                         be 'T' for transpose of the matrix or 'N' for 
//                         non-transpose
//
// - MatrixMult_XY(): Calculates the product between a mxk matrix A and a
//                    kxn matrix B and saves it in the mxn matrix C. X and Y
//                    may be 'T' or 'N' for transpose or non-transpose of A
//                    or B respectively
//
// - MatrixMult_NN_Sub(): Same as MatrixMult, but calculates A*B-C
//
// - MatrixTrans(): Calculates the transpose of a input matrix 
//                  and saves it in dst
//
// - Additional Note: The library is based on Fortran, which is column-major.
//                    However, C is row-major. Thus we need to transpose
//                    matrices before acting any routine on them. 
//
// - Documentation about each library routine can be found on netlib.
//
// *****************************************************************************

/*----IMPORT BLAS FORTRAN ROUTINES ----*/
extern void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N,
		   int* K, double* ALPHA, double* A, int* LDA,
		   double* B, int* LDB, double* BETA,
		   double* C, int* LDC);

extern double ddot_(int* N, double* DX, int* INCX, 
		    double* DY, int* INCY);
extern void dgemv_(char* TRANS, int* M, int* N, double* ALPHA, 
		   double* A, int* LDA, double* X, int* INCX, 
		   double* BETA, double* Y, int* INCY);
extern void dscal_(int* N, double* DA, double* DX, int* INCX);
extern void dcopy_(int* N, double* DX, int* INCX, 
		   double* DY, int* INCY);
extern void daxpy_(int* N, double* DA, double* DX, int* INCX, 
		   double* DY, int* INCY);
extern double dasum_(int* N, double* DX, int* INCX);


/*--------WHERE DO I PUT YOU?---------*/
void* xmalloc(size_t size);
void* xcalloc(size_t num, size_t size);

/*----------- BLAS LEVEL 1 ------------*/
double ASum(int n, double* vec);
void* SubVec(int n, double* vec1, double* vec2);
void* CopyVec(int n, double* vec, double* target);
void* ScaleVec(int n, double* vec, double alpha);
double DotProduct(int n, double* v1, double* v2);

/*----------- BLAS LEVEL 2 ------------*/
void* MatrixVectorMult_N(int m, int n, double* A,
			 double* x, double* out);
void* MatrixVectorMult_T(int n, int m, double* A, 
			 double* x, double* out);


/*----------- BLAS LEVEL 3 ------------*/
void* MatrixMult_NN(int m, int n, int k, double* A,
		    double* B, double* C);
void* MatrixMult_NN_Sub(int m, int n, int k, double* A,
			double* B, double* C);
void* MatrixMult_TN(int m, int n, int k, double* A,
		    double* B, double* C);
void* MatrixMult_NT(int m, int n, int k, double* A,
		    double* B, double* C);
void* MatrixMult_TT(int m, int n, int k, double* A,
		    double* B, double* C);
void* MatrixTrans(const int n, const int m,
		  double* src, double* dst);

/*-------------------------------------*/


void* xmalloc(size_t size) {
  void* ptr = malloc(size);
  if (ptr == NULL) {
    fprintf(stderr, "Out of memory\n");
    exit(1);
  }

  return ptr;
}

void* xcalloc(size_t num, size_t size) {
  void* ptr = calloc(num, size);
  if (ptr == NULL) {
    fprintf(stderr, "Out of memory\n");
    exit(1);
  }

  return ptr;
}

double ASum(int n, double* vec) {
  int N, INCX;
  double ret = 0.0;
  N = n;
  INCX = 1;
  ret = dasum_(&N, vec, &INCX);
  return ret;
}

void* SubVec(int n, double* vec1, double* vec2) {
  int N, INCX, INCY;
  double DA;
  INCX = 1;
  INCY = 1;
  N = n;
  DA = -1.0;

  daxpy_(&N, &DA, vec2, &INCX, vec1, &INCY);

  return NULL;
}
  

void* CopyVec(int n, double* vec, double* target) {
  int INCX, INCY, N;
  N = n;
  INCX = 1;
  INCY = 1;

  dcopy_(&N, vec, &INCX, target, &INCY);

  return NULL;
}

void* ScaleVec(int n, double* vec, double alpha) {
  int INCX, N;
  N = n;
  INCX = 1;
  dscal_(&N, &alpha, vec, &INCX);

  return NULL;
}

double DotProduct(int n, double* v1, double* v2) {
  double ret = 0.0;
  int N, INCX, INCY;
  INCX = 1;
  INCY = 1;
  N = n;
  ret = ddot_(&N, v1, &INCX, v2, &INCY);
  return ret;
}

void* MatrixVectorMult_N(int m, int n, double* A, double* x, double* out) {
  char TRANS;
  int M, N, LDA, INCX, INCY;
  double ALPHA, BETA;

  double* awork;
  awork = xmalloc(n*m*sizeof(double));

  MatrixTrans(m, n, A, awork);

  N = n;
  M = m;
  LDA = M;
  INCX = 1;
  INCY = 1;

  ALPHA = 1.0;
  BETA = 0.0;

  TRANS = 'N';

  dgemv_(&TRANS, &M, &N, &ALPHA, awork, &LDA, x, &INCX, &BETA, out, &INCY);

  free( awork );
  return NULL;
}

void* MatrixVectorMult_T(int n, int m, double* A, double* x, double* out) {
  char TRANS;
  int M, N, LDA, INCX, INCY;
  double ALPHA, BETA;

  double* awork;
  awork = xmalloc(n*m*sizeof(double));

  MatrixTrans(m, n, A, awork);
 
  N = n;
  M = m;
  LDA = M;
  INCX = 1;
  INCY = 1;

  ALPHA = 1.0;
  BETA = 0.0;

  TRANS = 'T';

  dgemv_(&TRANS, &M, &N, &ALPHA, awork, &LDA, x, &INCX, &BETA, out, &INCY);

  free( awork );

  return NULL;
}


void* MatrixMult_NN(int m, int n, int k, double* A, double* B, double* C) {
  int M, N, K, LDA, LDB, LDC;
  double ALPHA, BETA;
  char TRANSA, TRANSB;

  double *awork, *bwork, *cwork;
  awork = xmalloc(m*k*sizeof(double));
  bwork = xmalloc(n*k*sizeof(double));
  cwork = xmalloc(m*n*sizeof(double));

  MatrixTrans(m, k, A, awork);
  MatrixTrans(k, n, B, bwork);

  M = m;
  N = n;
  K = k;
  LDA = M;//K;
  LDB = K;//N;
  LDC = M;

  ALPHA = 1.0;
  BETA = 0.0;

  TRANSA = 'N';
  TRANSB = 'N';

  dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, awork, &LDA, bwork, &LDB, &BETA,
	 cwork, &LDC);

  MatrixTrans(n, m, cwork, C);

  free( awork );
  free( bwork );
  free( cwork );

  return NULL;
}

void* MatrixMult_NN_Sub(int m, int n, int k, double* A, double* B, double* C) {
  int M, N, K, LDA, LDB, LDC;
  double ALPHA, BETA;
  char TRANSA, TRANSB;

  double *awork, *bwork, *cwork;
  awork = xmalloc(m*k*sizeof(double));
  bwork = xmalloc(n*k*sizeof(double));
  cwork = xmalloc(m*n*sizeof(double));

  MatrixTrans(m, k, A, awork);
  MatrixTrans(k, n, B, bwork);
  MatrixTrans(m, n, C, cwork);

  M = m;
  N = n;
  K = k;

  LDA = M;
  LDB = K;
  LDC = M;

  ALPHA = 1.0;
  BETA = -1.0;

  TRANSA = 'N';
  TRANSB = 'N';

  dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, awork, &LDA, bwork, &LDB, &BETA,
	 cwork, &LDC);

  MatrixTrans(n, m, cwork, C);

  free( awork );
  free( bwork );
  free( cwork );

  return NULL;
}

void* MatrixMult_TN(int m, int n, int k, double* A, double* B, double* C) {
  int M, N, K, LDA, LDB, LDC;
  double ALPHA, BETA;
  char TRANSA, TRANSB;

  double *awork, *bwork, *cwork;
  awork = xmalloc(m*k*sizeof(double));
  bwork = xmalloc(n*k*sizeof(double));
  cwork = xmalloc(m*n*sizeof(double));
  MatrixTrans(k, m, A, awork);
  MatrixTrans(k, n, B, bwork);

  M = m;
  N = n;
  K = k;
  LDA = K;
  LDB = K;
  LDC = M;

  ALPHA = 1.0;
  BETA = 0.0;

  TRANSA = 'T';
  TRANSB = 'N';

  dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, awork, &LDA, bwork, &LDB, &BETA,
	 cwork, &LDC);

  MatrixTrans(n, m, cwork, C);

  free( awork );
  free( bwork );
  free( cwork );

  return NULL;
}

void* MatrixMult_NT(int m, int n, int k, double* A, double* B, double* C) {
  int M, N, K, LDA, LDB, LDC;
  double ALPHA, BETA;
  char TRANSA, TRANSB;

  double *awork, *bwork, *cwork;
  awork = xmalloc(m*k*sizeof(double));
  bwork = xmalloc(n*k*sizeof(double));
  cwork = xmalloc(m*n*sizeof(double));

  MatrixTrans(m, k, A, awork);
  MatrixTrans(n, k, B, bwork);

  M = m;
  N = n;
  K = k;
  LDA = M;
  LDB = N;
  LDC = M;

  ALPHA = 1.0;
  BETA = 0.0;

  TRANSA = 'N';
  TRANSB = 'T';

  dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, awork, &LDA, bwork, &LDB, &BETA,
	cwork, &LDC);

  MatrixTrans(n, m, cwork, C);
  
  free( awork );
  free( bwork );
  free( cwork );

  return NULL;
}


void* MatrixMult_TT(int n, int m, int k, double* A, double* B, double* C) {
  int M, N, K, LDA, LDB, LDC;
  double ALPHA, BETA;
  char TRANSA, TRANSB;

  double *awork, *bwork, *cwork;
  awork = xmalloc(m*k*sizeof(double));
  bwork = xmalloc(n*k*sizeof(double));
  cwork = xmalloc(m*n*sizeof(double));

  MatrixTrans(k, m, A, awork);
  MatrixTrans(n, k, B, bwork);

  M = m;
  N = n;
  K = k;

  LDA = K;
  LDB = N;
  LDC = M;
  ALPHA = 1.0;
  BETA = 0.0;

  TRANSA = 'T';
  TRANSB = 'T';

  dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, awork, &LDA, bwork, &LDB, &BETA,
	 cwork, &LDC);

  MatrixTrans(n, m, cwork, C);

  free( awork );
  free( bwork );
  free( cwork );

  return NULL;
}

void* MatrixTrans(const int rows, const int cols, double* src, double* dst) {
  //pragma omp parallel for
  //rows and cols correspond to src, not dst
  int i, j;
  for(int k = 0; k < rows*cols; k++) {
    i = k/rows;
    j = k%rows;
    *(dst+k) = *(src+cols*j+i);
  }

  return NULL;
}
