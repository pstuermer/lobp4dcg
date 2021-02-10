#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <omb.h>
#include <assert.h>
#include <errno.h>
#include <time.h>

#ifndef RAND_MAX
#define RAND_MAX((int) ((unsigned) ~0 >> 1))
#endif

void InitRandom(void);
double RandomReal(double low, double high);
void* InitEigV(size_t n, size_t num_eig, double *Z,
	       double low, double high);
void* GetSearchDir(size_t n, size_t num_eig, double* K,
		   double* M, double* X, double* Y,
 double* search_dir);
void* SetupPQ(size_t n, size_t num_eig, double* PQ, double *out,
	      double *KM, double *Z, double *search_dir, char mode);


void InitRandom(void) {
  srand((int) time(NULL));
}

double RandomReal(double low, double high) {
  double d;
  d = (double) rand() / ((double) RAND_MAX + 1);
  return (low + d* (high-low));
}

void* InitEigV(size_t n, size_t num_eig, double* Z,
	       double low, double high) {
  for(size_t i = 0; i < num_eig; i++){
    for(size_t j = 0; j < n; j++) {
      *(Z+j*num_eig+i) = RandomReal(low, high, RAND_MAX);
    }
  }
}

void* GetSearchDir(size_t n, size_t num_eig, double* K,
		   double *M, double* X, double* Y) {

  double num1, num2;
  double* work;

  /* Calculate the first rho as initial start for the calculation */
  MatrixMult_NN(n, 

void* SetupPQ(size_t n, size_t num_eig, double* PQ, double* out,
	      double* KM, double* X, double* Y,
	      double* search_dir, char mode) {  

  /* calculate X*diag(search_dir) or Y*diag(search_dir) */
  MatrixMult_NN(n, num_eig, num_eig, X, search_dir, PQ);

  /* calculate KX-Y*diag(search_dir) or MY-X*diag(search_dir) */
  MatrixMult_NN_Sub(n, num_eig, num_eig, KM, Y, PQ);
}


