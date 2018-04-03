#/usr/bin/bash

COMPILE_MODE     = release
#COMPILE_MODE     = debug
USE_COMPLEX      = 0
USE_OPENMP       = 0
USE_PROFILE      = 0
USE_PEXSI        = 0
USE_GPU          = 1

ifeq (${COMPILE_MODE}, release)
  COMPILE_DEF    = -DRELEASE
  COMPILE_FLAG   = -fast -w -D GPUDIRECT -no-ipo
  COMPILE_FLAG   = -fast -w -no-ipo
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
  CUDA_FLAG = -I/ccs/home/jiawl/DG/GPU/include -D GPU
  CUDA_LIB = -L${CUDA_DIR}/lib64 -lcufft -lcublas -lcudart -lcuda  -lmagma #-lmagma_sparse #-lcudadevrt
  NVCCFLAG = -gencode arch=compute_35,code=sm_35 -DGPU -I ../include -DSYNC  # for debug purpose
  NVCCFLAG = -gencode arch=compute_35,code=sm_35 -DGPU -I ../include 
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

  PEXSI_DIR        = /ccs/home/jiawl/DG/crd-pexsi
  DSUPERLU_DIR     = /ccs/home/jiawl/software/SuperLU_DIST_5.1.1
  METIS_DIR        = /ccs/home/jiawl/software/parmetis/lib
  SCOTCH_DIR       = 

  PEXSI_INCLUDE    = -I$(PEXSI_DIR)/include
  PEXSI_SRC_LIB    = ${PEXSI_DIR}/src/libpexsi_edison_v0.10.0.a
  DSUPERLU_LIB     = ${DSUPERLU_DIR}/lib/libsuperlu_dist5.1.1.a
  SCOTCH_LIB       = -L${SCOTCH_DIR} -lptscotchparmetis -lptscotch -lptscotcherr -lscotch
  METIS_LIB        = ${METIS_DIR}/libmetis.a
  PEXSI_LIB        = ${PEXSI_SRC_LIB} ${DSUPERLU_LIB} ${SCOTCH_LIB}  ${METIS_LIB} 
  PEXSI_LIB        = ${PEXSI_SRC_LIB} ${DSUPERLU_LIB} ${METIS_LIB} 
endif

DGDFT_DIR     = /ccs/home/jiawl/DG/pwdft/crd-dgdft

# inclues
BLOPEX_INCLUDE   = -I${DGDFT_DIR}/external/blopex
FFTW_INCLUDE     = -I/opt/cray/fftw/3.3.4.11/x86_64/include
DGDFT_INCLUDE    = -I${DGDFT_DIR}/include 
INCLUDES         = ${DGDFT_INCLUDE} ${PEXSI_INCLUDE} ${LIBXC_INCLUDE} ${BLOPEX_INCLUDE} ${FFTW_INCLUDE} ${MAGMA_INC}

# Libraries
LIBXC_DIR     = /ccs/home/jiawl/software/libxc
LIBXC_LIB        = $(LIBXC_DIR)/lib/libxc.a
LIBXC_INCLUDE    = -I$(LIBXC_DIR)/include
BLOPEX_LIB       = #${DGDFT_DIR}/external/blopex/libblopex.a 
FFTW_LIB         = -L/opt/cray/fftw/3.3.4.11/x86_64/lib -lfftw3_mpi -lfftw3 -lm
MKL_LIB          = -Wl,--start-group  ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm
MKL_LIB          = #-mkl=cluster

DGDFT_LIB    = ${DGDFT_DIR}/src/libdgdft.a
LIBS         = ${DGDFT_LIB} ${PEXSI_LIB} ${LIBXC_LIB} ${BLOPEX_LIB} ${FFTW_LIB} ${MKL_LIB} ${IPM} ${CUDA_LIB} ${MAGMA_LIB}

CC           = cc
CXX          = CC
FC           = ftn
LOADER       = CC

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
