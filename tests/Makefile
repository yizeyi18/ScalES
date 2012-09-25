# NOTE: This Makefile does NOT support auto-dependency for the .h files.
# If the header files are changed, do "make clean" first.

include ../make.inc.${HOSTID}

SRCS = ex17.cpp

#SRCS = ex1.cpp ex2.cpp ex3.cpp ex4.cpp ex5.cpp ex6.cpp ex7.cpp ex8.cpp\
			 ex9.cpp ex10.cpp ex11.cpp ex12.cpp ex13.cpp ex14.cpp ex15.cpp\
			 ex16.cpp

OBJS = ${SRCS:.cpp=.o}
DEPS = ${SRCS:.cpp=.d}
EXES = ${SRCS:.cpp=}	

ex17: ex17.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex17.o $(LOADOPTS) )

ex16: ex16.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex16.o $(LOADOPTS) )

ex15: ex15.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex15.o $(LOADOPTS) )

-include ${DEPS}


${DGDFT_LIB}:
	(cd ${DGDFT_DIR}/src; make all)

cleanlib:
	(${RM} -f ${DGDFT_LIB})

cleanall:
	rm -f ${EXES} ${OBJS} ${DEPS} *.d.o
