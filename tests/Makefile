# NOTE: This Makefile does NOT support auto-dependency for the .h files.
# If the header files are changed, do "make clean" first.

include ../make.inc.${HOSTID}

SRCS = ex1.cpp ex2.cpp ex3.cpp ex4.cpp
OBJS = ${SRCS:.cpp=.o}
DEPS = ${SRCS:.cpp=.d}

ex5: ex5.o ${DGDFTLIB} 
	($(LOADER) $(LOADOPTS) -o $@ ex5.o)

ex4: ex4.o ${DGDFTLIB} 
	($(LOADER) $(LOADOPTS) -o $@ ex4.o)

ex3: ex3.o ${DGDFTLIB} 
	($(LOADER) $(LOADOPTS) -o $@ ex3.o)

ex2: ex2.o ${DGDFTLIB} 
	($(LOADER) $(LOADOPTS) -o $@ ex2.o)

ex1: ex1.o ${DGDFTLIB} 
	($(LOADER) $(LOADOPTS) -o $@ ex1.o)


-include ${DEPS}


${DGDFTLIB}:
	(cd ${DGDFTDIR}/src; make libdgdft.a)


cleanlib:
	(${RM} -f ${DGDFTLIB})

cleanall:
	rm -f ex1 ex2 ex3 ex4 *.o *.d *.d.o
