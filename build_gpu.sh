#/bin/bash
module purge &&  module load cgpu cmake/3.18.2 git gcc cuda openmpi fftw
cmake -H. -Bbuild_gpu -DCMAKE_TOOLCHAIN_FILE=$PWD/config/toolchain.cmake.cori
make -C build_gpu -j10
export OMP_NUM_THREADS=1
srun -n 1 build_gpu/src/pwdft/pwdft -in examples/pwdft/01-ground_pbe/pwdft.in
