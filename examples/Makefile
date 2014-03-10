# NOTE: This Makefile does NOT support auto-dependency for the .h files.
# If the header files are changed, do "make clean" first.

include ../make.inc

SRCS = NVTdgdft_Update.cpp dgdft.cpp diagonalize.cpp

OBJS = ${SRCS:.cpp=.o}
DEPS = ${SRCS:.cpp=.d}
EXES = ${SRCS:.cpp=}	

NVTdgdft_Update: NVTdgdft_Update.o ${NVTdgdft_Update_LIB} 
	($(LOADER) -o $@ NVTdgdft_Update.o $(LOADOPTS) )

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
