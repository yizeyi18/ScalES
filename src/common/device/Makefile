include ../../../make.inc
SRCS = device_utility.cpp

OBJS = ${SRCS:.cpp=.o} ${SRCF90:.f90=.o} ${SRCF:.f=.o} 

all: device_utility.o

-include ${DEPS}

cleanall:
	${RM} -f ${OBJS} ${DEPS} *.d.*
