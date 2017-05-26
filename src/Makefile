# NOTE: This Makefile does NOT support auto-dependency for the .h files.
# If the header files are changed, do "make clean" first.

include ../make.inc

SRCS = scf_dg.cpp scf.cpp eigensolver.cpp  iondynamics.cpp \
			 hamiltonian_dg_matrix.cpp hamiltonian_dg.cpp hamiltonian.cpp \
			 periodtable.cpp spinor.cpp fourier.cpp utility.cpp esdf.cpp \
			 mpi_interf.cpp scalapack.cpp lapack.cpp blas.cpp global.cpp 

OBJS = ${SRCS:.cpp=.o} ${SRCF90:.f90=.o} ${SRCF:.f=.o}
DEPS = ${SRCS:.cpp=.d} ${SRCF90:.f90=.d} ${SRCF:.f=.d}

all: libdgdft.a

-include ${DEPS}

libdgdft.a: ${OBJS}
	$(AR) $(ARFLAGS) $@ ${OBJS}
	$(RANLIB) $@

cleanall:
	${RM} -f ${OBJS} ${DEPS} libdgdft.a *.d.*
