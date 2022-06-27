#/usr/bin/bash

COMPILE_MODE     = release
#COMPILE_MODE     = debug
USE_COMPLEX      = 0
USE_OPENMP       = 0
USE_PROFILE      = 0
USE_PEXSI        = 0
USE_GPU          = 0

ifeq (${COMPILE_MODE}, release)
  COMPILE_DEF    = -DRELEASE
#  COMPILE_FLAG   = -fast -w -D GPUDIRECT -no-ipo
  COMPILE_FLAG   = -O3 -w #-D GPU -D GPUDIRECT
endif
ifeq (${COMPILE_MODE}, debug)
  COMPILE_DEF    = -DDEBUG=1
  COMPILE_FLAG   = -O2 -w
endif

ifeq (${USE_COMPLEX}, 1)
  SCALAR_DEF    = -DCOMPLEXSCALAR -DCPX
endif

ifeq (${USE_GPU}, 1)
  NVCC = nvcc
  CUDA_FLAG = -I/usr/local/cuda/include -D GPU -I/usr/local/magma/include
  CUDA_LIB = -L/usr/local/cuda/lib64 -lcufft -lcublas -lcudart -lcuda #  -lmagma #-lmagma_sparse #-lcudadevrt
  CUDA_LIB += -L/usr/local/magma/lib -lmagma
  #NVCCFLAG = -gencode arch=compute_60,code=sm_60 -DGPU -I ../include -DSYNC  # for debug purpose
  NVCCFLAG = -gencode arch=compute_60,code=sm_60 -DGPU -I ../include  -I /usr/local/cuda/include
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

  PEXSI_DIR        = 
  DSUPERLU_DIR     = 
  METIS_DIR        = 
  SCOTCH_DIR       = 

  PEXSI_INCLUDE    = 
  PEXSI_SRC_LIB    = 
  DSUPERLU_LIB     = 
  SCOTCH_LIB       = 
  METIS_LIB        = 
  PEXSI_LIB        = 
  PEXSI_LIB        = 
endif

DGDFT_DIR     = /staff/qcjiang/codes/hefeidgdft
FFTW_DIR      = /staff/qcjiang/lib/fftw-3.3.10

FFTW_INCLUDE     = -I${FFTW_DIR}/include  
MKL_ROOT      = /staff/qcjiang/intel/oneapi/mkl/2022.0.2
# inclues
DGDFT_INCLUDE    = -I${DGDFT_DIR}/include 
INCLUDES         = ${DGDFT_INCLUDE} ${PEXSI_INCLUDE} ${LIBXC_INCLUDE} ${BLOPEX_INCLUDE} ${FFTW_INCLUDE} ${MAGMA_INC}

# Libraries

LIBXC_DIR     = /staff/qcjiang/lib/libxc-5.2.3
LIBXC_LIB        = $(LIBXC_DIR)/lib/libxc.a
LIBXC_INCLUDE    = -I$(LIBXC_DIR)/include
FFTW_LIB         = -L${FFTW_DIR}/lib -lfftw3_mpi -lfftw3 -lm
#MKL_LIB      = -L${MKL_ROOT}/lib/intel64 -lmkl_core -lmkl_intel_lp64  -lmkl_sequential -lmkl_blacs_intelmpi_lp64  -lmkl_scalapack_lp64 -lpthread -lgfortran 
MKL_LIB      = -Wl,--start-group ${MKL_ROOT}/lib/intel64/libmkl_sequential.a ${MKL_ROOT}/lib/intel64/libmkl_core.a ${MKL_ROOT}/lib/intel64/libmkl_blacs_openmpi_lp64.a ${MKL_ROOT}/lib/intel64/libmkl_scalapack_lp64.a ${MKL_ROOT}/lib/intel64/libmkl_intel_lp64.a -Wl,--end-group -lgfortran -ldl -lpthread -zmuldefs
#GFORTRAN_LIB     = /usr/lib/gcc/x86_64-linux-gnu/5/libgfortran.a
DGDFT_LIB    = ${DGDFT_DIR}/src/libdgdft.a
RQRCP_LIB    = ${DGDFT_DIR}/external/rqrcp/librqrcp.a
LBFGS_LIB    = ${DGDFT_DIR}/external/lbfgs/liblbfgs.a
LIBS         = ${DGDFT_LIB} ${PEXSI_LIB} ${LIBXC_LIB} ${GFORTRAN_LIB} ${BLOPEX_LIB} ${LBFGS_LIB} ${RQRCP_LIB} ${FFTW_LIB} ${MKL_LIB} ${IPM} ${CUDA_LIB} ${MAGMA_LIB}

CC           = mpicc
CXX          = mpicxx
FC           = mpif90
LOADER       = mpicxx

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
CXXFLAGS     = ${COMPILE_FLAG} ${OPENMP_FLAG} ${PROFILE_FLAG} ${INCLUDES} ${CUDA_FLAG} -std=c++11
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
	${FC} -c ${FFLAGS} $< ${MKL_ROOT}/lib/intel64/libmkl_intel_lp64.a 
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
