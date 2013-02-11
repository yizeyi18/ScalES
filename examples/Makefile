# NOTE: This Makefile does NOT support auto-dependency for the .h files.
# If the header files are changed, do "make clean" first.

include ../make.inc.${HOSTID}

SRCS = dgdft.cpp

OBJS = ${SRCS:.cpp=.o}
DEPS = ${SRCS:.cpp=.d}
EXES = ${SRCS:.cpp=}	

dgdft: dgdft.o ${DGDFT_LIB} 
	($(LOADER) -o $@ dgdft.o $(LOADOPTS) )

-include ${DEPS}

${DGDFT_LIB}:
	(cd ${DGDFT_DIR}/src; make all)

cleanlib:
	(${RM} -f ${DGDFT_LIB})

cleanall:
	rm -f ${EXES} ${OBJS} ${DEPS} *.d.o
