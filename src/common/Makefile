include ../../make.inc

SRCS = blas.cpp  fourier.cpp  global.cpp  lapack.cpp  mpi_interf.cpp  scalapack.cpp  utility.cpp

#OBJS = ${SRCS:.cpp=.o} ${SRCF90:.f90=.o} ${SRCF:.f=.o} 
OBJS = ${SRCS:.cpp=.o} ${SRCF90:.f90=.o} ${SRCF:.f=.o} ${SRCCU:.cu=.o}

ifeq (${DEVICE}, nvgpu)
	vpath %.cpp . device device/nvgpu
	vpath %.cu  device/nvgpu
	SRCS = blas.cpp  fourier.cpp  global.cpp  lapack.cpp  mpi_interf.cpp  scalapack.cpp  utility.cpp device_utility.cpp device_blas.cpp  device_error.cpp  device_fft.cpp  device_mpi_interf.cpp  device_solver.cpp 
	SRCCU = device_utils.cu
endif

all: libcommon.a

-include ${DEPS}

libcommon.a: ${OBJS} 
	$(AR) $(ARFLAGS) $@ ${OBJS}  
	$(RANLIB) $@

cleanall:
	${RM} -f ${OBJS}   ${DEPS} libcommon.a *.d.*
