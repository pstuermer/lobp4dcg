#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>
#include <tgmath.h>
#include <omp.h>
#include <cblas.h>

#include "blaswrapper.c"

#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

// Routines:
//
// - Pinverse(): Calculates the Pseudoinverse of a mxn matrix A and save it in B
//
// - CopyMatrix(): Copies a mxn matrix A to target
//
// - Get1Norm(): Calculates the 1-norm of a mxn matrix
//
// - GetQ(): Performs a QR-factorization of a mxn matrix A and then saves Q in A
//
// - SVD(): Performs a singular value decomposition of a mxn matrix A and saves
//          singular values in s. If needed vectors may also be computed and saved
//
// - CalcEigVal(): Calculates eigenvalues and righthand eigenvectors of a nxn matrix
//                 A and saves them in EigVal and EigVec respectively
//
// - MInverse(): Calculates the inverse of a mxm matrix A via LU-decomposition and
//               replaces the inverse with A
//
// - Additional Note: The library is based on Fortran, which is column-major.
//                    However, C is row-major. Thus we need to transpose
//                    matrices before acting any routine on them.
// 
// - Documentation about each library routine can be found on netlib.
//
// *************************************************************************************

/*---- IMPORT LAPACK FORTRAN ROUTINES ----*/
extern void dgetrf_(int *M, int *N, double *A, int *lda,
		    int *IPIV, int *INFO);

extern void dgetri_(int* N, double* A, int* lad, int* IPIV, 
		    double* WORK, int* lwork, int* INFO);

extern void dgeev_(char* JOBVL, char* JOBVR, int* N, double* A,
		   int* LDA, double* WR, double* WI, double* VL,
		   int* LDVL, double* VR, int* LDVR, double* WORK,
		   int* LWORK, int* INFO);

extern void dgesdd_(char* JOBZ, int* M, int* N, double* A, int* LDA,
		    double* S, double* U, int* LDU, double* VT,
		    int* LDVT, double* WORK, int* LWORK, int* IWORK,
		    int* INFO);
extern void dgeqrf_(int* M, int* N, double* A, int* LDA, double* TAU,
		    double* WORK, int* LWORK, int* INFO);

extern void dorgqr_(int* M, int* N, int* K, double* A, int* LDA,
		    double* TAU, double* WORK, int* LWORK, int* INFO);

extern void dormqr(char* SIDE, char* TRANS, int* M, int* N, int* K,
		   double* A, int* LDA, double* TAU, double* C,
		   int* LDC, double* WORK, int* LWORK, int* INFO);

extern double dlange_(char* NORM, int* M, int* N, double* A,
		      int* LDA, double* WORK);

extern void dlacpy_(char* UPLO, int* M, int* N, double* A, 
		    int* LDA, double* B, int* LDB);

extern void dgelss_(int* M, int* N, int* NRHS, double* A, 
		    int* LDA, double* B, int* LDB, double* S,
		    double* RCOND, int* RANK, double* WORK,
		    int* LWORK, int* INFO);

void* Pinverse(int m, int n, double* A, double* B);
void* CopyMatrix(int m, int n, double* A, double* target);
double Get1Norm(int m, int n, double* A);
void* GetQ(int m, int n, double* A);
void* SVD(int m, int n, double* s, double* A);
void* CalcEigVal(int n, double* A, double* EigVal, double* EigVec);
void* MInverse(double* A, int m);

void* Pinverse(int m, int n, double* A, double* B) {
  int M, N, NRHS, LDA, LDB, RANK, LWORK, INFO;
  double *WORK, *S;
  double RCOND, WOPT;

  double *transA, *transB;
  transA = xmalloc(n*m*sizeof(double));
  transB = xmalloc(m*m*sizeof(double));
  MatrixTrans(m, n, A, transA);
  MatrixTrans(m, m, B, transB);

  M = m;
  N = n;
  NRHS = M;
  LDA = M;
  LDB = max(M,N);
  RCOND = -1.0;
  RANK = -1;
  S = xmalloc(min(m,n)*sizeof(double));

  /* Query and allocate the  optimal workspace */
  LWORK = -1;
  WOPT = 1.0;
  dgelss_(&M, &N, &NRHS, transA, &LDA, transB, &LDB, S, &RCOND, &RANK, &WOPT, &LWORK, &INFO);
  
  /* Compute Pseudoinverse and save n B */
  LWORK = WOPT;
  WORK = xmalloc(WOPT*sizeof(double));
  dgelss_(&M, &N, &NRHS, transA, &LDA, transB, &LDB, S, &RCOND, &RANK, WORK, &LWORK, &INFO);

  MatrixTrans(n, m, transA, A);
  MatrixTrans(m, m, transB, B);

  free( transB );
  free( transA );
  free( WORK );

  return NULL;
}

void* CopyMatrix(int m, int n, double* A, double* target) {
  char UPLO;
  int M, N, LDA, LDB;
  //  double* awork;
  M = m;
  N = n;
  LDA = M;
  LDB = M;

  //awork = xmalloc(m*n*sizeof(double));
  //MatrixTrans(m, n, A, awork);

  UPLO = 'A';

  dlacpy_(&UPLO, &M, &N, A, &LDA, target, &LDB);

  //free( awork );

  return NULL;
}

double Get1Norm(int m, int n, double* A) {
  int M, N, LDA;
  double* WORK;
  char NORM;
  double ret;

  double* awork;
  awork = xmalloc(m*n*sizeof(double));
  MatrixTrans(m, n, A, awork);
  
  NORM = '1';
  M = m;
  N = n;
  LDA = M;
  WORK = xmalloc(M*sizeof(double));

  ret = dlange_(&NORM, &M, &N, awork, &LDA, WORK);

  free( WORK );
  return ret;
}

void* GetQ(int m, int n, double* A) {
  /* dgeqrf computes a QR factorization of a real M-by-N Matrix A */
  /*                                                              */
  /* dorgqr generates an M-by-N real matrix Q with orthonormal columns */
  /* which is defined as the first N columns of a product of K elementary */
  /* reflectors of order M Q = H(1)H(2)...H(k) */
  /* as returned by dgeqrf */

  int M, N, K, LDA, LWORK, INFO;
  double *TAU, *WORK;
  double WKOPT;
  //  char SIDE, TRANS;

  double *transA;//, *transC;
  transA = xmalloc(n*m*sizeof(double));
  //  transC = xmalloc(n*m*sizeof(double));
  MatrixTrans(m, n, A, transA);
  //  MatrixTrans(m, n, C, transC);

  INFO = 0;
  M = m;
  N = n;
  LDA = M;
  WKOPT = 0.0;
  TAU = xmalloc(min(M,N)*sizeof(double));

  /* Query and allocate the optimal workspace */
  LWORK = -1;
  dgeqrf_(&M, &N, transA, &LDA, TAU, &WKOPT, &LWORK, &INFO);

  LWORK = WKOPT;
  WORK = xmalloc(LWORK*sizeof(double));

  /* Get QR-factorization */
  dgeqrf_(&M, &N, transA, &LDA, TAU, WORK, &LWORK, &INFO);

  if (INFO < 0) {
    printf("The algorithm failed to compute the QR-factorization.\n");
    exit( 1 );
  }
  free( WORK );
   
  WKOPT = 0.0;
  K = min(M,N);
  INFO = 0;
  
  // Query and allocate the optimal workspace 
  LWORK = -1;
  dorgqr_(&M, &N, &K, transA, &LDA, TAU, &WKOPT, &LWORK, &INFO);

  LWORK = WKOPT;
  WORK = xmalloc(LWORK*sizeof(double));
  
  // Get Q
  dorgqr_(&M, &N, &K, transA, &LDA, TAU, WORK, &LWORK, &INFO);

  if (INFO < 0) {
    printf("The algorithm failed to extract Q of a QR-factorization.\n");
    exit( 1 );
  }

  MatrixTrans(n, m, transA, A);
  // MatrixTrans(n, m, transC, C);

  free( transA );
  //  free( transC );
  free( WORK );
  free( TAU );

  return NULL;
}

void* SVD(int m, int n, double* s, double* A) {
  int M, N, LDA, LDU, LDVT, INFO, LWORK;
  double WKOPT;
  double *WORK, *U, *VT;
  int* IWORK;
  char JOBZ;

  double* transA;
  transA = xmalloc(m*n*sizeof(double));
  MatrixTrans(m, n, A, transA);

  M = m;
  N = n;
  LDA = M;
  LDU = M;
  LDVT = N;
  INFO = 0;

  JOBZ = 'N';

  U = xmalloc(LDU*M*sizeof(double));
  VT = xmalloc(LDVT*N*sizeof(double));
  IWORK = xmalloc(8*min(m,n)*sizeof(int));

  /* Query and allocate the optimal workspace */

  LWORK = -1;
  dgesdd_(&JOBZ, &M, &N, transA, &LDA, s, U, &LDU, VT, &LDVT, &WKOPT,
	 &LWORK, IWORK, &INFO);

  LWORK = WKOPT;
  WORK = xmalloc(LWORK*sizeof(double));

  /* Compute the SVD */
  dgesdd_(&JOBZ, &M, &N, transA, &LDA, s, U, &LDU, VT, &LDVT, WORK,
	 &LWORK, IWORK, &INFO);

  if (INFO > 0) {
    printf("The algorithm computing SVD failed to converge.\n");
    exit( 1 );
  }

  MatrixTrans(n, m, transA, A);

  free( transA );
  free( U );
  free( VT );
  free( WORK );
  free( IWORK );

  return NULL;
}

void* CalcEigVal(int n, double* A, double* EigVal, double* EigVec) {
  int N, LDA, LDVL, LDVR, INFO, LWORK;
  double WKOPT;
  double* WORK;
  double *WR, *WI, *VL, *VR;
  char JOBVL, JOBVR;

  N = n;
  LDA = N;
  LDVL = N;
  LDVR = N;
  WR = xmalloc(N*sizeof(double));
  WI = xmalloc(N*sizeof(double));
  VL = xmalloc(N*LDVL*sizeof(double));
  VR = xmalloc(N*LDVR*sizeof(double));
  JOBVL = 'N';
  JOBVR = 'V';

  double* transA;
  transA = xmalloc(N*N*sizeof(double));
  MatrixTrans(n, n, A, transA);

  // Query and allocate the optimal workspace
  LWORK = -1;
  WKOPT = 0.0;
  dgeev_(&JOBVL, &JOBVR, &N, transA, &LDA, EigVal, WI, VL, &LDVL, VR,
	 &LDVR, &WKOPT, &LWORK, &INFO);

  // Calculate eigenvalues and eigenvectors
  LWORK = (int)WKOPT;
  WORK = xmalloc(LWORK*sizeof(double));
  dgeev_(&JOBVL, &JOBVR, &N, transA, &LDA, EigVal, WI, VL, &LDVL, VR,
	 &LDVR, WORK, &LWORK, &INFO);
  
  //  for(int i = 0; i < N; i++) {
  //    printf("%f+i*%f\t", *(EigVal+i),*(WI+i));
  //  }
  //  printf("\n");
  
  if (INFO > 0) {
    printf("The algorithm failed to compute eigenvalues.\n");
    exit( 1 );
  }

  MatrixTrans(n, n, transA, A);
  MatrixTrans(n, n, VR, EigVec);

  free( WR );
  free( WI );
  free( VL );
  free( VR );
  free( transA );
  free( WORK );
  return NULL;
}


void* MInverse(double* A, int m) {
  int LWORK, INFO, LDA;
  double *WORK;
  int *IPIV;
  double WKOPT;

  double* transA;
  transA = xmalloc(m*m*sizeof(double));
  MatrixTrans(m, m, A, transA);

  LDA = max(1,m);
  INFO = 0;
  IPIV = xmalloc(m*sizeof(int));
  WORK = xmalloc(LWORK*sizeof(double));

  // calculate LU-factorization of matrix
  dgetrf_(&m, &m, transA, &LDA, IPIV, &INFO);
  
  // initialize workspace for inverse calculation
  LWORK = -1;
  INFO = 0;
  WKOPT = 0.0;
  dgetri_(&m, transA, &LDA, IPIV, &WKOPT, &LWORK, &INFO);

  LWORK = WKOPT;
  free( WORK );
  WORK = xmalloc(LWORK*sizeof(double));

  // given a LU factorization in transA, calculate the inverse
  dgetri_(&m, transA, &LDA, IPIV, WORK, &LWORK, &INFO);

  MatrixTrans(m, m, transA, A);
 
  free( transA );
  free( IPIV );
  free( WORK );

  return NULL;
}
