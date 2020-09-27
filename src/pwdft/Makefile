include ../../make.inc

SRCS = eigensolver.cpp  hamiltonian.cpp  iondynamics.cpp  pwdft.cpp  scf.cpp  spinor.cpp

OBJS = ${SRCS:.cpp=.o} ${SRCF90:.f90=.o} ${SRCF:.f=.o} 

ifeq (${DEVICE}, nvgpu)
	vpath %.cpp . device 
	SRCS = eigensolver.cpp  hamiltonian.cpp  iondynamics.cpp  pwdft.cpp  scf.cpp  spinor.cpp device_eigensolver.cpp	device_hamiltonian.cpp	device_spinor.cpp
endif



all: pwdft

-include ${DEPS}

pwdft: ${OBJS} ${DGDFT_LIB}
	($(LOADER) -o $@ ${OBJS} $(LOADOPTS) )

${DGDFT_LIB}:
	(cd ${DGDFT_DIR}/src/common; make all )

cleanall:
	rm -f ${EXES} ${OBJS}  ${DGDFT_LIB} ${DEPS} *.d.o pwdft
	cd ../common;  make cleanall
