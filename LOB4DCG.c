#include <stdio.h>
#include <string.h>
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

void* GetSearchDir(int n, int num_eig, double* K,
		    double* M, double* X, double* Y,
		    double* rho);
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


void* GetResidualNorm(int m, int num_eig, double* K, double* M,
		       double* X, double* Y,
		       double* eigVal, double* resnorm);


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

void* GetSearchDir(int n, int num_eig, double* K,
		    double* M, double* X, double* Y,
		    double* rho) {

  double num1, num2, denom;
  double *work, *TransX, *TransY;


  work = xmalloc(n*sizeof(double));
  TransX = xmalloc(n*num_eig*sizeof(double));
  TransY = xmalloc(n*num_eig*sizeof(double));

  MatrixTrans(n, num_eig, X, TransX);
  MatrixTrans(n, num_eig, Y, TransY);

  /* Calculate the first rho as initial start for the calculation */
  /* Calculate M*x from x^T*M*x, then the dot product */
  for (int i = 0; i < num_eig; i++) {
    MatrixVectorMult_N(n, n, K, TransX+i*n, work);
    num1 = DotProduct(n, TransX+i*n, work);

    MatrixVectorMult_N(n, n, M, TransY+i*n, work);
    num2 = DotProduct(n, TransY+i*n, work);

  /* Calculate the denominator */
    denom = DotProduct(n, TransX+i*n, TransY+i*n);
    denom = 2*fabs(denom);

    *(rho+i) = (num1 + num2) / denom;
  }

  free( TransX );
  free( TransY );
  free( work );

  return NULL;
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

  // Set all entries in Hsr to 0 to get rid of
  // diagonalization from previous iteration
  memset(Hsr, 0, 2*2*m*m*sizeof(double));

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

  double TransEigVec[m*m];
  double TransEigVecSort[n*m];
  MatrixTrans(m, m, EigVec, TransEigVec);

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
      *(TransEigVecSort+j+i*m) = *(TransEigVec+j+arrSorted[i]*m);
    }
  }
  MatrixTrans(n, m, TransEigVecSort, EigVecSort);
  return NULL;
}

void* SplitEigVec(int n, int m, double* EigVec, double* Xsr, double* Ysr) {
  for (int i = 0; i < n*m; i++) {
    if (i < n*m/2) {
      *(Ysr+i) = *(EigVec+i);
    } else {
      *(Xsr+i-n*m/2) = *(EigVec+i);
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
    normx = ASum(n, Xtrans+i*n);
    normy = ASum(n, Ytrans+i*n);

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

void* GetResidualNorm(int m, int num_eig, double* K, double* M,
		       double* X, double* Y,
		       double* eigVal, double* resnorm) {
  double nominator, denominator, work, normH;
  double *work1, *work2;
  work1 = xmalloc(m*sizeof(double));
  work2 = xmalloc(m*sizeof(double));

  double *TransX, *TransY;
  TransX = xmalloc(m*num_eig*sizeof(double));
  TransY = xmalloc(m*num_eig*sizeof(double));

  MatrixTrans(m, num_eig, X, TransX);
  MatrixTrans(m, num_eig, Y, TransY);

  double* Z;
  Z = xmalloc(2*m*sizeof(double));
  /* calculate denominator first */
  work = Get1Norm(m, m, K);
  normH = Get1Norm(m, m, M);
  normH = max(work, normH);

  for (int i = 0; i < num_eig; i++) {
    // create Z vector
    for(int j = 0; j < 2*m; j++) {
      if (j < m) {
	*(Z+j) = *(TransY+j+i*m);
      } else {
	*(Z+j) =*(TransX+j+i*m-m);
      }
    }
    // calculate denominator
    denominator = normH + eigVal[i];
    work = ASum(2*m, Z);
    denominator *= work;

  /* calculate nominator */
    MatrixVectorMult_N(m, m, K, TransX+i*m, work1);
    CopyVec(m, TransY+i*m, work2);
    ScaleVec(m, work2, eigVal[i]);
    SubVec(m, work1, work2);
    nominator = ASum(m, work1);

    MatrixVectorMult_N(m, m, M, TransY+i*m, work1);
    CopyVec(m, TransX+i*m, work2);
    ScaleVec(m, work2, eigVal[i]);
    SubVec(m, work1, work2);
    nominator += ASum(m, work1);

    resnorm[i] = nominator/denominator;
  }

  free( work1 );
  free( work2 );
  return NULL;
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
      printf("%.6e\t", *(Arr+j+i*row));
    }
    printf("\n");
  }
  printf("\n");

  return NULL;
}


int main(void) {
  int size = 1862;
  int numeig = 2;

  // Get off-diagonal matrices
    double *K, *M;
  K = xmalloc(size*size*sizeof(double));
  M = xmalloc(size*size*sizeof(double));

  LoadMatrix(K, size*size, "K.txt");
  LoadMatrix(M, size*size, "M.txt");
  /*
  double K[64] = {4.928320, 0.025793, 0.052951, -0.061457, 0.003689, 0.117414, -0.032168,-0.020979,
		  0.025793, 4.983653, -0.053805, -0.087340, 0.047473, -0.040380, 0.032443, 0.127336,
		  0.052951, -0.053805, 4.925665, 0.072362, -0.028482, 0.019990, 0.061028, -0.026909,
		  -0.061457, -0.087340, 0.072362, 4.817192, 0.076393, 0.044845, 0.014307, 0.092048,
		  0.003689, 0.047473, -0.028482, 0.076393, 4.906923, 0.049800, -0.006088, -0.121430,
		  0.117414, -0.040380, 0.019990, 0.044845, 0.049800, 4.938623, -0.097788, 0.027653,
		  -0.032168, 0.032443, 0.061028, 0.014307, -0.006088, -0.097788, 4.938039, 0.046206,
		  -0.020979, 0.127336, -0.026909, 0.092048, -0.121430, 0.027653, 0.046206, 4.842173};

  double M[64] = {4.771923, -0.078584, -0.048857, -0.030294, -0.011211, -0.016098, 0.066103, 0.009089,
		  -0.078584, 4.885744, 0.009801, 0.095645, 0.019762, 0.045331, -0.009861, 0.004445,
		  -0.048857, 0.009801, 4.808262, 0.016009, 0.065539, -0.107984, 0.022994, 0.102522,
		  -0.030294, 0.095645, 0.016009, 4.894399, -0.010249, -0.067917, 0.007982, 0.079571,
		  -0.011211, 0.019762, 0.065539, -0.010249, 4.730263, 0.022011, 0.016681, -0.004333,
		  -0.016098, 0.045331, -0.107984, -0.067917, 0.022011, 4.874459, 0.035306, -0.059140,
		  0.066103, -0.009861, 0.022994, 0.007982, 0.016681, 0.035306, 4.760353, -0.008420,
		  0.009089, 0.004445, 0.102522, 0.079571, -0.004333, -0.059140, -0.008420, 4.681968};
  */
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
  

  //  GetQ(size, numeig, X);
  //  GetQ(size, numeig, Y);
  //  NormalizeEigVec(size, numeig, X, Y);
  /*double X[16] = {0.680375, -0.444451,
		  -0.211234, 0.107940,
		  0.566198, -0.045206,
		  0.596880, 0.257742,
		  0.823295, -0.270431,
		  -0.604897, 0.026802,
		  -0.329554, 0.904459,
		  -0.536459, 0.832390};
  double Y[16] = {0.271423, -0.686642,
		  0.434594, -0.198111,
		  -0.716795, -0.740419,
		  0.213938, -0.782382,
		  -0.967399, 0.997849,
		  -0.514226, -0.563486,
		  -0.725537, 0.025865,
		  0.608354, 0.678224};*/
  // Initialize first search direction
  double rho0[numeig];
  GetSearchDir(size, numeig, K, M, X, Y, rho0);

  double *rho;
  rho = xcalloc(numeig*numeig, sizeof(double));
  for (int i = 0; i < numeig; i++) {
    *(rho+i+i*numeig) = rho0[i];
  }

  double res1[numeig];
  GetResidualNorm(size, numeig, K, M, X, Y, rho0, res1);

  // Setup P and Q
  double *P, *Q;
  P = xmalloc(size*numeig*sizeof(double));
  Q = xmalloc(size*numeig*sizeof(double));
  SetupPQ(size, numeig, P, K, Y, X, rho);
  SetupPQ(size, numeig, Q, M, X, Y, rho);

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
  double* Winv;
  Winv = xmalloc(2*numeig*2*numeig*sizeof(double));
  SetupW(size, 2*numeig, U, V, Winv);
  MInverse(Winv, 2*numeig);
  double s[2*numeig];
  SVD(2*numeig, 2*numeig, s, Winv);
  //  PrintArray(2*numeig, 1, s);
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
  CalcEigVal(4*numeig, Hsr, eVal, eVec);

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
  GetResidualNorm(size, numeig, K, M, X1, Y1, eValSort, res);
  PrintArray(numeig, 1, res);

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
  Hsr = calloc(6*numeig*6*numeig,sizeof(double));
  Xsr = xmalloc(numeig*3*numeig*sizeof(double));
  Ysr = xmalloc(numeig*3*numeig*sizeof(double));
  eVec = xmalloc(6*numeig*6*numeig*sizeof(double));
  eVecSort = xmalloc(numeig*6*numeig*sizeof(double));

  double eVal1[6*numeig];

  double* eVecwork;
  eVecwork = xmalloc(size*numeig*sizeof(double));
  double* singval;
  singval = xmalloc(3*numeig*sizeof(double));
 
  int maxiter = 200;
  for(int iter = 0; iter < maxiter; iter++) {

    // setup new P & Q
    SetupPQ(size, numeig, P, K, Y1, X1, rho);
    SetupPQ(size, numeig, Q, M, X1, Y1, rho);

    // Apply Precondittioner
    ApplyGenericPreconditioner(size, numeig, Kinv, P);
    ApplyGenericPreconditioner(size, numeig, Minv, Q);

    // Setup U and V
    SetupUV(size, numeig, P, X1, X, U);
    SetupUV(size, numeig, Q, Y1, Y, V);
    
    // Orthogonalize U and V
    GetQ(size, 3*numeig, U);
    GetQ(size, 3*numeig, V);

    // Calculate W and inverse it
    SetupW(size, 3*numeig, U, V, Winv);
    MInverse(Winv, 3*numeig);

    // Setup Hsr
    SetupHsr(size, 3*numeig, U, V, Winv, K, M, Hsr);

    // Compute Eigenvalues and right side Eigenvectors of Hsr
    CalcEigVal(6*numeig, Hsr, eVal1, eVec);

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
    GetResidualNorm(size, numeig, K, M, X, Y, eValSort, res);
    printf("Print residual error and eigenvalues at iteration %d", iter);
    printf("\n");
    PrintArray(numeig, 1, res);
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
  free( singval );
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
