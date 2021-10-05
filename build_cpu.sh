#! /bin/bash
# old with Make
# cd ../src; make -j 12; cd ../examples; rm $1; rm $1.o; make $1 
# new with cmake
BDIR=build_scales_cpu
rm -rf ${BDIR}
/global/common/sw/cray/cnl7/haswell/cmake/3.18.2/bin/cmake -H. -B${BDIR} -DCMAKE_TOOLCHAIN_FILE=/global/homes/w/weihu/project_weihu/jielanli/ScalES/scales_test/config/toolchain.cmake.cori \
  -DCMAKE_BUILD_TYPE=Release -DScalES_ENABLE_CUDA=OFF -DScalES_ENABLE_OPENMP=OFF -Duse_openmp=OFF
make -C ${BDIR} -j 12
