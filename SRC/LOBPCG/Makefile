include ../../make.inc.${HOSTID}

HEADERS = \
interpreter.h \
multivector.h \
lobpcg.h \
fortran_matrix.h \
fortran_interpreter.h \
fortran_options.h

LIB = libBLOPEX.a

##################################################################
# Targets
##################################################################

all: $(LIB) 

objects = fortran_matrix.o multivector.o lobpcg.o 

fortran_matrix.o: fortran_matrix.c fortran_matrix.h

multivector.o: multivector.c multivector.h

lobpcg.o: lobpcg.c lobpcg.h fortran_matrix.h

clean: 
	rm -f *.o *.a
	

##################################################################
# Rules
##################################################################

$(LIB) : $(objects)
	@echo  "Building $@ ... "
	echo $(objects)
	$(AR) $(ARFLAGS) $@ $(objects)
	$(RANLIB) $@
