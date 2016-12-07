#/usr/bin/bash

COMPILE_MODE     = release
#COMPILE_MODE     = debug
USE_COMPLEX      = 0
USE_OPENMP       = 1
USE_PROFILE      = 0
USE_PEXSI        = 1
USE_GPU          = 1

ifeq (${COMPILE_MODE}, release)
  COMPILE_DEF    = -DRELEASE
  COMPILE_FLAG   = -O3 -w 
endif
ifeq (${COMPILE_MODE}, debug)
  COMPILE_DEF    = -DDEBUG=1
  COMPILE_FLAG   = -O2 -w
endif

ifeq (${USE_COMPLEX}, 1)
  SCALAR_DEF    = -DCOMPLEXSCALAR
endif

ifeq (${USE_GPU}, 1)
  NVCC = nvcc
  CUDA_FLAG = -D GPU -D GPUDIRECT
  CUDA_FLAG = -D GPU 
  CUDA_DIR = /usr/local/cuda
  CUDA_INCLUDE  = -I${CUDA_DIR}/include
  MAGMA_DIR = /opt/magma/2.1.0/mkl-gcc
  MAGMA_INCLUDE = -I${MAGMA_DIR}/include 
  MAGMA_LIB = -L${MAGMA_DIR}/lib
  CUDA_LIB = -L${CUDA_DIR}/lib64 -lcufft -lcublas -lcudart -lcuda ${MAGMA_LIB} -lmagma -lmagma_sparse #-lcudadevrt
  NVCCFLAG = -gencode arch=compute_52,code=sm_52 -DGPU -I ../include -D SYNC
endif

ifeq (${USE_OPENMP}, 1)
  OPENMP_DEF   = -DOPENMP
	OPENMP_FLAG  = -fopenmp
endif

ifeq (${USE_PROFILE}, 1)
	PROFILE_FLAG  = -g -pg
endif

ifeq (${USE_PEXSI}, 1)
  PEXSI_DEF        = -DPEXSI

  PEXSI_DIR        = /home/weile/PEXSI/pexsi
  DSUPERLU_DIR     = /home/weile/PEXSI/SuperLU_DIST_5.1.1
  METIS_DIR        = /home/weile/PEXSI/parmetis/lib
  SCOTCH_DIR       = 

  PEXSI_INCLUDE    = -I$(PEXSI_DIR)/include
  PEXSI_SRC_LIB    = #${PEXSI_DIR}/src/libpexsi_edison_v0.10.0.a
  DSUPERLU_LIB     = ${DSUPERLU_DIR}/lib/libsuperlu_dist.a
  SCOTCH_LIB       = -L${SCOTCH_DIR} -lptscotchparmetis -lptscotch -lptscotcherr -lscotch
  METIS_LIB        = ${METIS_DIR}/libmetis.a
  PEXSI_LIB        = ${PEXSI_SRC_LIB} ${DSUPERLU_LIB} ${SCOTCH_LIB}  ${METIS_LIB} 
  PEXSI_LIB        = ${PEXSI_SRC_LIB} ${DSUPERLU_LIB} ${METIS_LIB} 
endif

DGDFT_DIR     = /home/weile/test

# inclues
BLOPEX_INCLUDE   = -I${DGDFT_DIR}/external/blopex
FFTW_INCLUDE     = -I/home/weile/software/fftw3/include
DGDFT_INCLUDE    = -I${DGDFT_DIR}/include 
LIBXC_INCLUDE    = -I/opt/libxc/2.1.2/intel/include
INCLUDES         = ${DGDFT_INCLUDE} ${PEXSI_INCLUDE} ${LIBXC_INCLUDE} ${BLOPEX_INCLUDE} ${FFTW_INCLUDE} ${CUDA_INCLUDE} ${MAGMA_INCLUDE}

# Libraries
LIBXC_LIB        = /opt/libxc/2.1.2/intel/lib/libxc.a
BLOPEX_LIB       = #${DGDFT_DIR}/external/blopex/libblopex.a 
FFTW_LIB         = #-L/opt/fftw/3.3.0.1/x86_64/lib -lfftw3_mpi -lfftw3 -lm
FFTW_LIB         = -L/home/weile/software/fftw3/lib -lfftw3_mpi -lfftw3 -lm
#MKL_LIB          = -Wl,--start-group  ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm
MKL_LIB          = -mkl=cluster 
DGDFT_LIB    = ${DGDFT_DIR}/src/libdgdft.a
LIBS         = ${DGDFT_LIB} ${PEXSI_LIB} ${LIBXC_LIB} ${BLOPEX_LIB} ${FFTW_LIB} ${MKL_LIB} ${IPM} ${CUDA_LIB} ${MAGMA_LIB}

CC           = mpicc
CXX          = mpicc
FC           = mpif90
LOADER       = mpicc

AR           = ar
ARFLAGS      = rvcu
# For System V based machine without ranlib, like Cray and SGI,
# use touch instead.
#RANLIB      = touch
RANLIB       = ranlib

RM           = rm
RMFLAGS      = -f

# Different compiling and linking options.
#


CFLAGS       = ${COMPILE_FLAG} ${OPENMP_FLAG} ${PROFILE_FLAG} ${INCLUDES} ${CUDA_FLAG}
FFLAGS       = ${COMPILE_FLAG} ${OPENMP_FLAG} ${PROFILE_FLAG} ${INCLUDES} ${CUDA_FLAG}
CXXFLAGS     = ${COMPILE_FLAG} ${OPENMP_FLAG} ${PROFILE_FLAG} ${INCLUDES} ${CUDA_FLAG}
CCDEFS       = ${COMPILE_DEF} ${PEXSI_DEF} ${SCALAR_DEF} ${OPENMP_DEF}
CPPDEFS      = ${COMPILE_DEF} ${PEXSI_DEF} ${SCALAR_DEF} ${OPENMP_DEF}
LOADOPTS     = ${OPENMP_FLAG} ${PROFILE_FLAG} ${LIBS} 


# Compilation replacement rules

%.o: %.c
	${CC} -c ${CFLAGS} ${CCDEFS} $< 
%.o: %.cpp
	${CXX} -c ${CXXFLAGS} ${CPPDEFS} $< 
%.o: %.f
	${FC} -c ${FFLAGS} $<
%.o: %.F
	${FC} -c ${FFLAGS} $<
%.o: %.f90
	${FC} -c ${FFLAGS} $<
%.o: %.F90
	${FC} -c ${FFLAGS} $<
%.o: %.cu
	$(NVCC) -c ${NVCCFLAG} $<

# Generate auto-dependencies (for cpp files now)
%.d: %.cpp
	@set -e; rm -f $@; \
	$(CXX) -M $(CXXFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@;\
	rm -f $@.$$$$
