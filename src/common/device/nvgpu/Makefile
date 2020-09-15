include ../../../../make.inc
SRCS = device_blas.cpp  device_error.cpp  device_fft.cpp  device_mpi_interf.cpp  device_solver.cpp

OBJS = ${SRCS:.cpp=.o} ${SRCF90:.f90=.o} ${SRCF:.f=.o} ${SRCCU:.cu=.o}


all: device_blas.o  device_error.o  device_fft.o  device_mpi_interf.o  device_solver.o


cleanall:
	${RM} -f ${OBJS} ${DEPS} libcommon.a *.d.*
