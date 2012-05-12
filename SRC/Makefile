# LLIN: NOTE: READ LOBPCG/Makefile.inc before migrating this DIR to
# other places!
#
include ../make.inc.${HOSTID}

# With SCFPW version
OBJS 	= esdfutil.o  esdf.o  vecmatop.o spline.o parallel.o periodtable.o \
	lobpcg.o eigpw.o  scfpw.o lglnodes.o interp.o scfdg_setup.o scfdg_scf.o \
	eigdg.o eigdg_A.o eigdg_std.o 

# NO SCFPW version
#OBJS 	= esdfutil.o  esdf.o  multi_vector.o  \
	pcg_multi.o vecmatop.o spline.o parallel.o periodtable.o \
	eigpw.o  \
	lglnodes.o interp.o scfdg_setup.o scfdg_scf.o eigdg.o eigdg_A.o eigdg_std.o

DEP     = $(OBJS:.o=.d)


libdft.a: ${OBJS}
	$(AR) $(ARFLAGS) $@ $(OBJS)
	$(RANLIB) $@

mdpw: 	mdpw.o libdft.a $(LOBPCG_LIB) 
	$(LOADER) $(LOADOPTS) -o mdpw mdpw.o $(OBJS) $(ALIBS) $(IPM)

mddg: 	mddg.o libdft.a $(LOBPCG_LIB) 
	$(LOADER) $(LOADOPTS) -o mddg mddg.o $(OBJS) $(ALIBS) $(IPM)

tstpw: 	tstpw.o libdft.a $(LOBPCG_LIB) 
	$(LOADER) $(LOADOPTS) -o tstpw tstpw.o $(OBJS) $(ALIBS)

tstdg: 	tstdg.o libdft.a $(LOBPCG_LIB)
	$(LOADER) $(LOADOPTS) -o tstdg tstdg.o libdft.a $(ALIBS)

$(LOBPCG_LIB):
	(cd LOBPCG; make)



testlobpcg: testlobpcg.o libdft.a $(LOBPCG_LIB)
	$(LOADER) $(LOADOPTS) -o testlobpcg testlobpcg.o $(OBJS) $(ALIBS)

testeig: pdeig.o testeig.o
	$(LOADER) testeig.o pdeig.o $(PARPACK) -o testeig

testeig1: testeig1.o
	$(LOADER) testeig1.o $(PARPACK) -o testeig1

testeig2: testeig2.o
	$(LOADER) testeig2.o $(PARPACK) -o testeig2

ttmatvec1: ttmatvec1.o
	$(LOADER) $(LOADOPTS) -o ttmatvec1 ttmatvec1.o 

testpqr: testpqr.o vecmatop.o
	$(LOADER) $(LOADOPTS) -o $@ $^

testpge: testpge.o
	$(LOADER) $(LOADOPTS) -o $@ $<

testge: testge.o
	$(LOADER) $(LOADOPTS) -o $@ $<

convcube: convcube.o
	$(LOADER) $(LOADOPTS) -o $@ $<

clean:
	(cd LOBPCG; make clean; cd ../; $(RM) $(RMFLAGS) *.o *.d *.a tstpw tstdg mdpw mddg \
	  testeig1 testeig ttmatvec1 testeig2 testpge testge)

help:
	echo "usage: make [ dgdft | clean ]"


-include $(DEP)
