# NOTE: This Makefile does NOT support auto-dependency for the .h files.
# If the header files are changed, do "make clean" first.

include ../make.inc

SRCS = dgdftNVE.cpp dgdftNH1.cpp dgdftNH2.cpp dgdft.cpp diagonalize.cpp

OBJS = ${SRCS:.cpp=.o}
DEPS = ${SRCS:.cpp=.d}
EXES = ${SRCS:.cpp=}	

dgdftNVE: dgdftNVE.o ${dgdftNVE_LIB} 
	($(LOADER) -o $@ dgdftNVE.o $(LOADOPTS) )

dgdftNH1: dgdftNH1.o ${dgdftNH1_LIB} 
	($(LOADER) -o $@ dgdftNH1.o $(LOADOPTS) )

dgdftNH2: dgdftNH2.o ${dgdftNH2_LIB} 
	($(LOADER) -o $@ dgdftNH2.o $(LOADOPTS) )

dgdft: dgdft.o ${DGDFT_LIB} 
	($(LOADER) -o $@ dgdft.o $(LOADOPTS) )

diagonalize: diagonalize.o ${diagonalize_LIB} 
	($(LOADER) -o $@ diagonalize.o $(LOADOPTS) )


-include ${DEPS}

${DGDFT_LIB}:
	(cd ${DGDFT_DIR}/src; make all)

cleanlib:
	(${RM} -f ${DGDFT_LIB})

cleanall:
	rm -f ${EXES} ${OBJS} ${DEPS} *.d.o
