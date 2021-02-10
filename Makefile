ARPACK_PATH = ../../ARPACK
BLAS_PATH = ../../ARPACK/BLAS
LAPACK_PATH = ../../ARPACK/LAPACK
#MKL_PATH = /opt/intel/composer_xe_2011_sp1.7.256/mkl/include
#MKL_FLAGS = -Wl,--start-group /opt/intel/composer_xe_2011_sp1.7.256/mkl/lib/intel64/libmkl_intel_lp64.a /opt/intel/composer_xe_2011_sp1.7.256/mkl/lib/intel64/libmkl_core.a /opt/intel/composer_xe_2011_sp1.7.256/mkl/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm -lifcore 


bdg-temp: testblas.c
	gcc -std=c11 -O2 -fopenmp -pedantic -Wall -Wextra testblas.c -o out1 -L$(ARPACK_PATH) -larpack_LINUX -L$(BLAS_PATH) -lblas -L$(LAPACK_PATH) -llapack -lpthread -lm -lgfortran -lgfortranbegin -lnsl

bdg-temp1: bdg-temp.c
	icc -std=c99 -O2 -openmp -Wextra bdg-temp.c -o out1 -I$(MKL_PATH) -lm -L$(ARPACK_PATH) -larpack_LINUX $(MKL_FLAGS)

blaswrapper: blaswrapper.c
	gcc -std=c11 -O2 -fopenmp -pedantic -Wall -Wextra blaswrapper.c -o blaswrapper -L$(BLAS_PATH) -lblas -L$(LAPACK_PATH) -llapack -lpthread -lm -lgfortran -lgfortranbegin -lnsl

lapackwrapper: lapackwrapper.c
	gcc -std=c11 -O2 -fopenmp -pedantic -Wall -Wextra lapackwrapper.c -o lapackwrapper -L$(BLAS_PATH) -lblas -L$(LAPACK_PATH) -llapack -lpthread -lm -lgfortran -lgfortranbegin -lnsl

LOB4DCG: LOB4DCG.c
	gcc -std=c11 -O2 -fopenmp -pedantic -Wall -Wextra LOB4DCG.c -o LOB4DCG -L$(BLAS_PATH) -lblas -L$(LAPACK_PATH) -llapack -lpthread -lm -lgfortran -lgfortranbegin -lnsl
