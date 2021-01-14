#! /bin/bash
# old with Make
# cd ../src; make -j 12; cd ../examples; rm $1; rm $1.o; make $1 
# new with cmake
BDIR=build_dgdft_cpu
rm -rf ${BDIR}
/home/linlin/Software/cmake-3.19.2-Linux-x86_64/bin/cmake -H. -B${BDIR} -DCMAKE_TOOLCHAIN_FILE=/home/linlin/Projects/dgdft/config/toolchain.cmake.linux \
  -DCMAKE_BUILD_TYPE=Release -DDGDFT_ENABLE_CUDA=OFF -DDGDFT_ENABLE_OPENMP=OFF
make -C ${BDIR} -j 12
