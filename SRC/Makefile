# NOTE: This Makefile does NOT support auto-dependency for the .h files.
# If the header files are changed, do "make clean" first.

include ../make.inc.${HOSTID}

SRCS = esdf.cpp global.cpp
OBJS = ${SRCS:.cpp=.o}
DEPS = ${SRCS:.cpp=.d}

lib: libdgdft.a


-include ${DEPS}

libdgdft.a: ${OBJS}
	$(AR) $(ARFLAGS) $@ ${OBJS}
	$(RANLIB) $@

cleanall:
	${RM} -f *.o libdgdft.a *.d *.d.*
