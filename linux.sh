#! /bin/bash
# old with Make
# cd ../src; make -j 12; cd ../examples; rm $1; rm $1.o; make $1 
# new with cmake
CMAKE=/home/yizeyi18/.local/bin/cmake
BDIR=a_build
rm -rf ${BDIR}
$CMAKE -H. \
 -B${BDIR} -DCMAKE_TOOLCHAIN_FILE="/home/yizeyi18/soft/src/scales/config/toolchain.cmake.yi" \
 -DCMAKE_CXX_FLAGS="-g" \
 -DCMAKE_C_FLAGS="-g" \
 -DCMAKE_Fortran_FLAGS="-g" -Wno-dev \
 -DCMAKE_INSTALL_PREFIX="/home/yizeyi18/soft/src/scales/install" -DScalES_ENABLE_PEXSI=ON \
 -DCMAKE_BUILD_TYPE=Release -DScalES_ENABLE_CUDA=OFF -DScalES_ENABLE_OPENMP=ON -DScalES_ENABLE_COMPLEX=OFF \
 -DScalES_ENABLE_MAGMA=OFF
#make -C ${BDIR} -j 12
$CMAKE --build ${BDIR}
$CMAKE --install ${BDIR}
