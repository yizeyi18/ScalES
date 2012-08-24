# NOTE: This Makefile does NOT support auto-dependency for the .h files.
# If the header files are changed, do "make clean" first.

include ../make.inc.${HOSTID}


SRCS = ex1.cpp ex2.cpp ex3.cpp ex4.cpp ex5.cpp ex6.cpp ex7.cpp ex8.cpp\
			 ex9.cpp ex10.cpp 
OBJS = ${SRCS:.cpp=.o}
DEPS = ${SRCS:.cpp=.d}
EXES = ${SRCS:.cpp=}	

ex10: ex10.o ${DGDFT_LIB} 
	($(LOADER) $(LOADOPTS) -o $@ ex10.o)

ex9: ex9.o ${DGDFT_LIB} 
	($(LOADER) $(LOADOPTS) -o $@ ex9.o)

ex8: ex8.o ${DGDFT_LIB} 
	($(LOADER) $(LOADOPTS) -o $@ ex8.o)

ex7: ex7.o ${DGDFT_LIB} 
	($(LOADER) $(LOADOPTS) -o $@ ex7.o)

ex6: ex6.o ${DGDFT_LIB} 
	($(LOADER) $(LOADOPTS) -o $@ ex6.o)

ex5: ex5.o ${DGDFT_LIB} 
	($(LOADER) $(LOADOPTS) -o $@ ex5.o)

ex4: ex4.o ${DGDFT_LIB} 
	($(LOADER) $(LOADOPTS) -o $@ ex4.o)

ex3: ex3.o ${DGDFT_LIB} 
	($(LOADER) $(LOADOPTS) -o $@ ex3.o)

ex2: ex2.o ${DGDFT_LIB} 
	($(LOADER) $(LOADOPTS) -o $@ ex2.o)

ex1: ex1.o ${DGDFT_LIB} 
	($(LOADER) $(LOADOPTS) -o $@ ex1.o)


-include ${DEPS}


${DGDFT_LIB}:
	(cd ${DGDFT_DIR}/src; make all)

cleanlib:
	(${RM} -f ${DGDFT_LIB})

cleanall:
	rm -f ${EXES} ${OBJS} ${DEPS} *.d.o
