# NOTE: This Makefile does NOT support auto-dependency for the .h files.
# If the header files are changed, do "make clean" first.

include ../make.inc

SRCS = ex45.cpp ex44.cpp ex43.cpp ex42.cpp ex41.cpp ex40.cpp ex39.cpp ex38.cpp ex37.cpp ex36.cpp \
	   ex35.cpp ex34.cpp ex31.cpp ex30.cpp ex29.cpp ex28.cpp ex27.cpp ex26.cpp \
	   ex25.cpp ex24.cpp ex23.cpp ex21.cpp ex22.cpp 

#SRCS = ex1.cpp ex2.cpp ex3.cpp ex4.cpp ex5.cpp ex6.cpp ex7.cpp ex8.cpp\
			 ex9.cpp ex10.cpp ex11.cpp ex12.cpp ex13.cpp ex14.cpp ex15.cpp\
			 ex16.cpp

OBJS = ${SRCS:.cpp=.o}
DEPS = ${SRCS:.cpp=.d}
EXES = ${SRCS:.cpp=}	

ex45: ex45.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex45.o $(LOADOPTS) )

ex44: ex44.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex44.o $(LOADOPTS) )

ex43: ex43.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex43.o $(LOADOPTS) )

ex42: ex42.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex42.o $(LOADOPTS) )

ex41: ex41.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex41.o $(LOADOPTS) )

ex40: ex40.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex40.o $(LOADOPTS) )

ex39: ex39.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex39.o $(LOADOPTS) )

ex38: ex38.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex38.o $(LOADOPTS) )

ex37: ex37.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex37.o $(LOADOPTS) )

ex36: ex36.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex36.o $(LOADOPTS) )

ex35: ex35.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex35.o $(LOADOPTS) )

ex34: ex34.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex34.o $(LOADOPTS) )

ex31: ex31.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex31.o $(LOADOPTS) )

ex30: ex30.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex30.o $(LOADOPTS) )

ex29: ex29.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex29.o $(LOADOPTS) )


ex28: ex28.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex28.o $(LOADOPTS) )

ex27: ex27.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex27.o $(LOADOPTS) )


ex26: ex26.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex26.o $(LOADOPTS) )


ex25: ex25.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex25.o $(LOADOPTS) )

ex24: ex24.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex24.o $(LOADOPTS) )

ex23: ex23.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex23.o $(LOADOPTS) )

ex22: ex22.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex22.o $(LOADOPTS) )

ex21: ex21.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex21.o $(LOADOPTS) )

ex20: ex20.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex20.o $(LOADOPTS) )

ex19: ex19.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex19.o $(LOADOPTS) )

ex18: ex18.o ${DGDFT_LIB} 
	($(LOADER) -o $@ ex18.o $(LOADOPTS) )

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
