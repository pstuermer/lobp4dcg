#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <omp.h>
#include <assert.h>
#include <errno.h>
#include <time.h>

#include "lapackwrapper.c"

#ifndef RAND_MAX
#define RAND_MAX((int) ((unsigned) ~0 >> 1))
#endif
void* LoadMatrix(double* M, int size, char filename[]);
void InitRandom(void);
double RandomReal(double low, double high);
void* InitEigV(int n, int num_eig, double *Z,
	       double low, double high);

double GetSearchDir(int n, double* K, double* M,
		   double* X, double* Y);
void* SetupPQ(int n, int num_eig, double* PQ,
	      double* KM, double* XY, double* YX, 
	      double* search_dir);
void* SetupUV1(int n, int num_eig, double* PQ,
	       double* XY, double* UV);
void* SetupUV(int n, int num_eig, double* PQ, double* XY,
	      double* XY_1, double* UV);
void* SetupW(int n, int m, double* U, double* V, double* W);
void* SetupHsr(int n, int m, double* U, double* V, double* W,
	       double* K, double* M, double* Hsr);
void* SortEig(int n, int m, double* EigVal, double* EigVec,
	      double EigValSort[n], double* EigVecSort);
void* SplitEigVec(int n, int m, double* EigVec, double* Xsr, double* Ysr);
void* ComputeEigVec(int n, int m, int num_eig,
		    double* XY, double* UV,double* W, 
		    double* EigVec, char mode);
void* NormalizeEigVec(int n, int num_eig, double* X, double* Y);


double GetResidualNorm(int m, double* K, double* M, double* X,
		       double* Y, double eigVal);


void* ApplyGenericPreconditionier(int m, int num_eig,
				  double* KMinv, double* PQ);


void* SingSetupKM(int n, int l12, double* Qinv, double* UV,
		  double* KM, double* out, char mode);
void* SingSplitKM(int r, int l12, double* KM, double* KM11,
		  double* KM12, double* KM22);
void* SingSetupCKM(int r, int l12, double* CKM, double* KM11,
		   double* KM12, double* KM22);
void* SingSetupHsr(int r, double* CK, double* CM, double* R,
		   double* Hsr);
void* SingSetupEigVec(int r, int l12, double* R, double* KM22,
		      double* KM12, double* EigVec, double* kernel,
		      double* out, char mode);


void* PrintArray(int row, int col, double* Arr);


void* LoadMatrix(double* M, int size, char filename[]) {
  double buffer = 0.0;
  int counter = 0;

  FILE* instream = fopen(filename, "r");
  if (instream) {
    while (fscanf(instream, "%lf", &buffer) && counter < size) {
      *(M+counter) = buffer;
      counter++;
    }
    fclose(instream);
  } else {
    /* Provide some error diagnostic */
    fprintf(stderr, "Could not open %s: ", filename);
    perror(0);
    errno = 0;
  }

  return NULL;
}

void InitRandom(void) {
  srand((int) time(NULL));
}

double RandomReal(double low, double high) {
  double d;
  d = (double) rand() / ((double) RAND_MAX + 1);
  return (low + d * (high-low));
}

void* InitEigV(int n, int num_eig, double* Z,
	       double low, double high) {
  for(int i = 0; i < num_eig; i++){
    for(int j = 0; j < n; j++) {
      *(Z+j*num_eig+i) = RandomReal(low, high);
    }
  }

  return NULL;
}

double GetSearchDir(int n, double* K,
		   double* M, double* X, double* Y) {

  double num1, num2, denom, ret;
  double *work, *x, *y;

  work = xmalloc(n*sizeof(double));
  x = xmalloc(n*sizeof(double));
  y = xmalloc(n*sizeof(double));


  /* Why is this in here? */
  for(int i = 0; i < n; i++) {
    *(x+i) = *(X+i);
    *(y+i) = *(Y+i);
  }

  /* Calculate the first rho as initial start for the calculation */
  /* Calculate M*x from x^T*M*x, then the dot product */
  MatrixVectorMult_N(n, n, K, x, work);
  num1 = DotProduct(n, x, work);

  MatrixVectorMult_N(n, n, M, y, work);
  num2 = DotProduct(n, y, work);

  /* Calculate the denominator */
  denom = DotProduct(n, x, y);
  denom = 2*fabs(denom);

  free( work );
  free( x );
  free( y );

  ret = (num1+num2)/denom;

  return ret;
}

void* SetupPQ(int n, int num_eig, double* PQ,
	      double* KM, double* YX, double* XY,
	      double* search_dir) {  

  /* calculate X*diag(search_dir) or Y*diag(search_dir) */
  MatrixMult_NN(n, num_eig, num_eig, YX, search_dir, PQ);

  /* calculate KX-Y*diag(search_dir) or MY-X*diag(search_dir) */
  MatrixMult_NN_Sub(n, num_eig, n, KM, XY, PQ);

  return NULL;
}


void* SetupUV1(int n, int num_eig, double* PQ, double* XY, double* UV) {
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < num_eig*2; j++) {
      if(j < num_eig) {
	*(UV+i*num_eig*2+j) = *(XY+j+i*num_eig);
      } else {
	*(UV+i*num_eig*2+j) = *(PQ+(j-num_eig)+i*num_eig);
      }
    }
  }

  return NULL;
}

void* SetupUV(int n, int num_eig, double* PQ, double* XY,
	      double* XY_1, double* UV) {
  for (int i = 0; i < n; i++) {
    for(int j = 0; j < num_eig*3; j++) {
      if(j < num_eig) {
	*(UV+i*num_eig*3+j) = *(XY+j+i*num_eig);
      } else if(j < 2*num_eig && j >= num_eig) {
	*(UV+i*num_eig*3+j) = *(XY_1+(j-num_eig)+i*num_eig);
      } else {
	*(UV+i*num_eig*3+j) = *(PQ+(j-2*num_eig)+i*num_eig);
      }
    }
  }

  return NULL;
}


void* SetupW(int n, int num_eig, double* U, double* V, double* W) {
  MatrixMult_TN(num_eig, num_eig, n, U, V, W);

  return NULL;
}

void* SetupHsr(int n, int m, double* U, double* V, double* W,
	       double* K, double* M, double* Hsr) {
  double *work, *work1;
  work = xmalloc(m*n*sizeof(double));
  work1 = xmalloc(m*m*sizeof(double));

  /* Compute top right block matrix first */
  MatrixMult_NN(n, m, n, K, U, work);
  MatrixMult_TN(m, m, n, U, work, work1);
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < m; j++) {
      *(Hsr+j+m*(2*i+1)) = *(work1+i+j*m);
    }
  }

  /* Compute bottom left block matrix */
  /* Assumes that W is inv(W) */
  double *work2, *work3;
  work2 = xmalloc(m*n*sizeof(double));
  work3 = xmalloc(m*m*sizeof(double));
  MatrixMult_NN(n, m, m, V, W, work);
  MatrixMult_NN(n, m, n, M, work, work2);
  MatrixMult_TN(m, m, n, V, work2, work3);
  MatrixMult_TN(m, m, m, W, work3, work1);
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < m; j++) {
      *(Hsr+2*m*m+j+2*m*i) = *(work1+i+j*m);
    }
  }

  free( work );
  free( work1 );
  free( work2 );
  free( work3 );

  return NULL;
}

void* SortEig(int n, int m, double* EigVal, double* EigVec,
	     double EigValSort[n], double* EigVecSort) {

  int arrSorted[n];
  double curr = 0.0;
  double last_min = 0.0;
  int idx = 0;

  for (int i = 0; i < n; i++) {
    curr = INFINITY;
    for (int j = 0; j < m; j++) {
      if (EigVal[j] <= last_min) {
	continue;
      } else {
	if (EigVal[j] < curr) {
	  curr = EigVal[j];
	  idx = j;
	} else {
	  continue;
	}
      }
    }
    arrSorted[i] = idx;
    EigValSort[i] = curr;
    last_min = curr;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      *(EigVecSort+j+i*m) = *(EigVec+j+arrSorted[i]*m);
    }
  }

  return NULL;
}

void* SplitEigVec(int n, int m, double* EigVec, double* Xsr, double* Ysr) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m/2; j++) {
      *(Ysr+j+i*m/2) = *(EigVec+j+i*m);
      *(Xsr+j+i*m/2) = *(EigVec+j+i*m+m/2);
    }
  }

  return NULL;
}

void* ComputeEigVec(int n, int m, int num_eig, double* XY, double* UV,
		    double* W, double* EigVec, char mode) {

  double *work;
  work = xmalloc(m*num_eig*sizeof(double));
  if ( mode == 'X') {
    MatrixMult_NN(n, num_eig, m, UV, EigVec, XY);
  } else if ( mode == 'Y') {
    MatrixMult_NN(m, num_eig, m, W, EigVec, work);
    MatrixMult_NN(n, num_eig, m, UV, work, XY);
  }

  free( work );

  return NULL;
}

void* NormalizeEigVec(int n, int num_eig, double* X, double* Y) {
  double *Xtrans, *Ytrans;
  Xtrans = xmalloc(n*num_eig*sizeof(double));
  Ytrans = xmalloc(n*num_eig*sizeof(double));

  MatrixTrans(n, num_eig, X, Xtrans);
  MatrixTrans(n, num_eig, Y, Ytrans);

  double norm, normx, normy;
  for(int i = 0; i < num_eig; i++) {
    normx = DotProduct(n, Xtrans+i*n, Xtrans+i*n);
    normy = DotProduct(n, Ytrans+i*n, Ytrans+i*n);

    norm = normx + normy;

    ScaleVec(n, Xtrans+i*n, 1.0/norm);
    ScaleVec(n, Ytrans+i*n, 1.0/norm);
  }

  MatrixTrans(num_eig, n, Xtrans, X);
  MatrixTrans(num_eig, n, Ytrans, Y);

  free( Xtrans );
  free( Ytrans);

  return NULL;
}      

double GetResidualNorm(int m, double* K, double* M, double* X, double* Y,
		      double eigVal) {
  double norm, nominator, denominator, work;
  double *work1, *work2;
  work1 = xmalloc(m*sizeof(double));
  work2 = xmalloc(m*sizeof(double));

  /* calculate denominator first */
  work = Get1Norm(m, m, K);
  denominator = Get1Norm(m, m, M);
  denominator = max(work, denominator);
  denominator += eigVal;
  work = DotProduct(m, X, X);
  work += DotProduct(m, Y, Y);
  denominator *= work;

  /* calculate nominator */
  MatrixVectorMult_N(m, m, K, X, work1);
  CopyVec(m, Y, work2);
  ScaleVec(m, work2, eigVal);
  SubVec(m, work1, work2);
  nominator = ASum(m, work1);

  MatrixVectorMult_N(m, m, M, Y, work1);
  CopyVec(m, X, work2);
  ScaleVec(m, work2, eigVal);
  SubVec(m, work1, work2);
  nominator += ASum(m, work1);

  norm = nominator/denominator;

  free( work1 );
  free( work2 );
  return norm;
}

void* ApplyGenericPreconditioner(int m, int num_eig, double* KMinv, double* PQ) {
  double* work;
  work = xmalloc(m*num_eig*sizeof(double));

  MatrixMult_NN(m, num_eig, m, KMinv, PQ, work);

  return work;
  free( work );
}

void* SingSetupKM(int n, int l12, double* Qinv, double* UV,
		  double* KM, double* out, char mode) {

  double *work1, *work2, *work3;
  work1 = xmalloc(n*l12*sizeof(double));
  work2 = xmalloc(n*l12*sizeof(double));
  work3 = xmalloc(l12*l12*sizeof(double));
  if (mode == 'K') {
    MatrixMult_NN(n, l12, n, KM, UV, work1);
    MatrixMult_TN(l12, l12, n/2, UV, work1, out);
  } else if (mode == 'M') {
    MatrixMult_NT(n, l12, l12, UV, Qinv, work1);
    MatrixMult_NN(n, l12, n, KM, work1, work2);
    MatrixMult_TN(l12, l12, n, UV, work2, work3);
    MatrixMult_NN(l12, l12, l12, Qinv, work3, out);
  }
  free( work1 );
  free( work2 );
  free( work3 );

  return NULL;
}


void* SingSplitKM(int r, int l12, double* KM, double* KM11,
		  double* KM12, double* KM22) {
  for (int i = 0; i < l12; i++) {
    for (int j = 0; j < l12; j++) {
      if (j < r && i < r) {
	*(KM11+j+i*r) = *(KM+j+i*l12);
      } else if (j >= r && i < r) {
	*(KM12+(j-r)+i*(l12-r)) = *(KM+j+i*l12);
      } else if (j >= r && i >= r) {
	*(KM22+(j-r)+(i-r)*(l12-r)) = *(KM+j+i*l12);
      }
    }
  }

  return NULL;
}


void* SingSetupCKM(int r, int l12, double* CKM, double* KM11,
		   double* KM12, double* KM22) {
  double *work1, *work2;
  work1 = xmalloc((l12-r)*r*sizeof(double));
  work2 = xmalloc(r*r*sizeof(double));

  MatrixMult_NT(l12-r, r, l12-r, KM22, KM12, work1);
  MatrixMult_NN(r, r, l12-r, KM12, work1, work2);

  for (int i = 0; i < r; i++) {
    for (int j = 0; j < r; j++) {
      *(CKM+j+i*r) = *(KM11+j+i*r)-*(work2+j+i*r);
    }
  }
  free( work1 );
  free( work2 );

  return NULL;
}

void* SingSetupHsr(int r, double* CK, double* CM, double* R, double* Hsr) {
  double *work1, *work2;
  work1 = xmalloc(r*r*sizeof(double));
  work2 = xmalloc(r*r*sizeof(double));

  /* Compute top right block matrix first */
  for(int i = 0; i < r; i++) {
    for(int j = 0; j < r; j++) {
      *(Hsr+j+r*(2*i+1)) = *(CK+j+i*r);
    }
  }

  /* Compute bottom left block matrix */
  MatrixMult_NT(r, r, r, CM, R, work1);
  MatrixMult_NN(r, r, r, R, work1, work2);
  for(int i = 0; i < r; i++) {
    for(int j = 0; j < r; j++) {
      *(Hsr+2*r*r+j+2*r*i) = *(work2+j+i*r);
    }
  }

  free( work1 );
  free( work2 );

  return NULL;
}

void* SingSetupEigVec(int r, int l12, double* R,
		      double* KM22, double* KM12, double* EigVec,
		      double* kernel, double* out, char mode) {
  double *work1, *work2, *work3;
  work1 = xmalloc(r*sizeof(double));
  work2 = xmalloc((l12-r)*sizeof(double));
  work3 = xmalloc((l12-r)*sizeof(double));
  
  if (mode == 'x') {
    MatrixVectorMult_T(l12-r, r, KM12, EigVec, work2);
    MatrixVectorMult_N(l12-r, l12-r, KM22, work2, work3);

    for(int i = 0; i < l12 - r; i++) {
      *(work2+i) = -*(work3+i)+*(kernel+i);
    }
    
    for(int i = 0; i < l12; i++) {
      if (i < r) {
	*(out+i) = *(EigVec+i);
      } else {
	*(out+i) = *(work2+i-r);
      }
    }
  } else if (mode == 'y') {
    MatrixVectorMult_T(r, r, R, EigVec, work1);
    MatrixVectorMult_T(l12-r, r, KM12, work1, work2);
    MatrixVectorMult_N(l12-r, l12-r, KM22, work2, work3);
    for(int i = 0; i < l12 - r; i++) {
      *(work2+i) = -*(work3+i) + *(kernel+i);
    }
    for(int i = 0; i < l12; i++) {
      if (i < r) {
	*(out+i) = *(work1+i);
      } else {
	*(out+i) = *(work2+i-r);
      }
    }
  }

  free( work1 );
  free( work2 );
  free( work3 );

  return NULL;
}

void* PrintArray(int row, int col, double* Arr) {
  for(int i = 0; i < col; i++) {
    for(int j = 0; j < row; j++) {
      printf("%f\t", *(Arr+j+i*row));
    }
    printf("\n");
  }
  printf("\n");

  return NULL;
}


int main(void) {
  int size = 1862;
  int numeig = 1;

  // Get off-diagonal matrices
  double *K, *M;
  K = xmalloc(size*size*sizeof(double));
  M = xmalloc(size*size*sizeof(double));

  LoadMatrix(K, size*size, "K.txt");
  LoadMatrix(M, size*size, "M.txt");

  // Get inverse for those, for preconditioner
  double *Kinv, *Minv;
  Kinv = xmalloc(size*size*sizeof(double));
  Minv = xmalloc(size*size*sizeof(double));

  CopyMatrix(size, size, K, Kinv);
  CopyMatrix(size, size, M, Minv);
  MInverse(Kinv, size);
  MInverse(Minv, size);

  // Initialize numeig random eigenvectors
  double *X, *Y;
  X = xmalloc(size*numeig*sizeof(double));
  Y = xmalloc(size*numeig*sizeof(double));

  InitEigV(size, numeig, X, -1.0, 1.0);
  InitEigV(size, numeig, Y, -1.0, 1.0);

  // Initialize first search direction
  double rho0 = GetSearchDir(size, K, M, X, Y);

  double *rho;
  rho = xcalloc(numeig*numeig, sizeof(double));
  for (int i = 0; i < numeig; i++) {
    *(rho+i+i*numeig) = rho0;
  }

  double res1[numeig];
  for (int i = 0; i < numeig; i++) {
    res1[i] = GetResidualNorm(size, K, M, X, Y, rho0);
    printf("%f\n", res1[i]);
  }
  printf("\n");

  // Setup P and Q
  double *P, *Q;
  P = xmalloc(size*numeig*sizeof(double));
  Q = xmalloc(size*numeig*sizeof(double));
  SetupPQ(size, numeig, P, K, X, Y, rho);
  SetupPQ(size, numeig, Q, M, Y, X, rho);

  //Apply Preconditioner, note that afterwards, P and Q are swapped
  ApplyGenericPreconditioner(size, numeig, Kinv, P);
  ApplyGenericPreconditioner(size, numeig, Minv, Q);

  // Setup U and V for i = 0
  double *U, *V;
  U = xmalloc(size*numeig*2*sizeof(double));
  V = xmalloc(size*numeig*2*sizeof(double));
  SetupUV1(size, numeig, P, X, U);
  SetupUV1(size, numeig, Q, Y, V);

  // Orthogonalize columns of U and V
  GetQ(size, 2*numeig, U);
  GetQ(size, 2*numeig, V);

  // Setup W
  /*  double *W;
  W = xmalloc(2*numeig*2*numeig*sizeof(double));
  SetupW(size, 2*numeig, U, V, W);*/

  double* Winv;
  Winv = xmalloc(2*numeig*2*numeig*sizeof(double));
  SetupW(size, 2*numeig, U, V, Winv);
  MInverse(Winv, 2*numeig);

  // Setup Hsr
  double* Hsr;
  Hsr = xcalloc(4*numeig*4*numeig, sizeof(double));
  SetupHsr(size, 2*numeig, U, V, Winv, K, M, Hsr);
  
  // Compute Eigenvalues and right side Eigenvectors of Hsr
  double eVal[4*numeig];
  double* eVec;
  eVec = xmalloc(4*numeig*4*numeig*sizeof(double));
  EigVal(4*numeig, Hsr, eVal, eVec);

  // Sort Eigenvalues and pick the numeig smallest positive ones
  // and their corresponding Eigenvectors
  double eValSort[numeig];
  double* eVecSort;
  eVecSort = xmalloc(numeig*4*numeig*sizeof(double));
  SortEig(numeig, 4*numeig, eVal, eVec, eValSort, eVecSort);

  // Split EigenVectors into X and Y component
  double *Xsr, *Ysr;
  Xsr = xmalloc(numeig*2*numeig*sizeof(double));
  Ysr = xmalloc(numeig*2*numeig*sizeof(double));
  SplitEigVec(numeig, 4*numeig, eVecSort, Xsr, Ysr);

  // go from Hsr EigenVectors to full Eigenvectors
  double *X1, *Y1;
  X1 = xmalloc(numeig*size*sizeof(double));
  Y1 = xmalloc(numeig*size*sizeof(double));
  ComputeEigVec(size, 2*numeig, numeig, X1, U, Winv, Xsr, 'X');
  ComputeEigVec(size, 2*numeig, numeig, Y1, V, Winv, Ysr, 'Y');

  // Normalize Eigenvectors
  NormalizeEigVec(size, numeig, X1, Y1);

  // Calculate residual norm
  double res[numeig];
  for (int i = 0; i < numeig; i++) {
    res[i] = GetResidualNorm(size, K, M, X1+i*size, Y1+i*size, eValSort[i]);
  }
  PrintArray(numeig, 1, res);
  PrintArray(numeig, 1, eValSort);

  // Assign Eigenvalues as new search directions (rho)
  for (int i = 0; i < numeig; i++) {
    *(rho+i+i*numeig) = *(eValSort+i);
  }

  free( U );
  free( V );
  free( Winv );
  free( Hsr );
  free( Xsr );
  free( Ysr );
  free( eVec );
  free( eVecSort);
  U = xmalloc(size*3*numeig*sizeof(double));
  V = xmalloc(size*3*numeig*sizeof(double));
  Winv = xmalloc(3*numeig*3*numeig*sizeof(double));
  Hsr = xmalloc(6*numeig*6*numeig*sizeof(double));
  Xsr = xmalloc(numeig*3*numeig*sizeof(double));
  Ysr = xmalloc(numeig*3*numeig*sizeof(double));
  eVec = xmalloc(6*numeig*6*numeig*sizeof(double));
  eVecSort = xmalloc(numeig*6*numeig*sizeof(double));

  double eVal1[6*numeig];

  double* eVecwork;
  eVecwork = xmalloc(size*numeig*sizeof(double));
 
  int maxiter = 100;
  for(int iter = 0; iter < maxiter; iter++) {

    // setup new P & Q
    SetupPQ(size, numeig, P, K, X1, Y1, rho);
    SetupPQ(size, numeig, Q, M, Y1, X1, rho);

    // Apply Precondittioner
    ApplyGenericPreconditioner(size, numeig, Kinv, P);
    ApplyGenericPreconditioner(size, numeig, Minv, Q);

    // Setup U and V
    SetupUV(size, numeig, Q, X1, X, U);
    SetupUV(size, numeig, P, Y1, Y, V);

    // Orthogonalize U and V
    GetQ(size, 3*numeig, U);
    GetQ(size, 3*numeig, V);

    // Calculate W and inverse it
    SetupW(size, 3*numeig, U, V, Winv);
    MInverse(Winv, 3*numeig);

    // Setup Hsr
    SetupHsr(size, 3*numeig, U, V, Winv, K, M, Hsr);

    // Compute Eigenvalues and right side Eigenvectors of Hsr
    EigVal(6*numeig, Hsr, eVal1, eVec);

    // Sort Eigenvalues and pick the numeig smallest positive ones
    // and their corresponding eigenvectors
    SortEig(numeig, 6*numeig, eVal1, eVec, eValSort, eVecSort);

    // Split Eigenvectors into X and Y component
    SplitEigVec(numeig, 6*numeig, eVecSort, Xsr, Ysr);

    // go from Hsr Eigenvectors to full Eigenvectors
    ComputeEigVec(size, 3*numeig, numeig, X, U, Winv, Xsr, 'X');
    ComputeEigVec(size, 3*numeig, numeig, Y, V, Winv, Ysr, 'Y');

    // Normalize Eigenvectors
    NormalizeEigVec(size, numeig, X, Y);

    // Calculate residual norm
    for (int i = 0; i < numeig; i++) {
      res[i] = GetResidualNorm(size, K, M, X+i*size, Y+i*size, eValSort[i]);
    }
    //    PrintArray(numeig, 1, res);
    PrintArray(numeig, 1, eValSort);

    // Assign Eigenvalues as new search directions
    for (int i = 0; i < numeig; i++) {
      *(rho+i+i*numeig) = *(eValSort+i);
    }

    // Assign Z(iter - 1) and Z(iter)
    CopyMatrix(size, numeig, X, eVecwork);
    CopyMatrix(size, numeig, X1, X);
    CopyMatrix(size, numeig, eVecwork, X1);

    CopyMatrix(size, numeig, Y, eVecwork);
    CopyMatrix(size, numeig, Y1, Y);
    CopyMatrix(size, numeig, eVecwork, Y1);
  }

  free( K );
  free( M );
  free( Kinv );
  free( Minv );
  free( X );
  free( Y );
  free( rho );
  free( P );
  free( Q );
  free( U );
  free( V );
  free( Hsr );
  free( Winv );
  free( eVec );
  free( eVecSort );
  free( Xsr );
  free( Ysr );
  free( X1 );
  free( Y1 );
  free( eVecwork);
  return 0;
}
