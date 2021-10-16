Scalable Electronic Structure (ScalES)
=====================================

Installation       
============

ScalES is written mostly in the C++ programming language with message passing interface (MPI), so you need to install MPI for parallel calculations. ScalES relies on some external numerical libraries (BLAS, LibXC, FFTW, LAPACK, BLACS, ScaLAPACK).

Requirements:
LibXC 5.x https://www.tddft.org/programs/libxc
FFTW 3.3.8 or later https://www.fftw.org/
blaspp https://bitbucket.org/icl/blaspp/src/master/
lapackpp https://bitbucket.org/icl/lapackpp/src/master/

ScalES can be installed using cmake (recommended) or the traditional make system.

Installation using cmake
------------------------

In order to compiler ScalES using cmake, you need edit a toolchain.cmake file. You can copy an example of `toolchain.cmake.cori` under the `config/` directory.

    cd ${ScalES_DIR}/config

Edit the variable in e.g. `toolchain.cmake.cori`.

Add the C-compiler, C++-compiler and fortran-compiler to use.
Add the directories for BLAS_PREFIX, ScaLAPACK, FFTW3_PREFIX and Libxc_ROOT.

Then edit the `${ScalES_DIR}/build_cpu.sh` script and modify the directories for cmake and toolchain.cmake

	./build_cpu.sh

You can find the executable file `pwdft` under `${ScalES_DIR}/build_scales_cpu/src` directory.


Installation using make
----------------------

In order to compiler ScalES using make, you need edit the `make.inc` file, which controls configuration of ScalES. You can copy an example of make.inc under the `config/` directory. We recommend to use the intel MKL and MPI libraries. For example,

    cd ${ScalES_DIR}
    cp config/make.inc.linux.intel make.inc

Edit the variables in make.inc

    ScalES_DIR  = Main directory for ScalES
    FFTW_DIR = directory for FFTW library
    MKL_ROOT = directory for MKL library
    BLASPP_DIR = directory for blaspp library
    LAPACKPP_DIR  = directory for lapackpp library
    LIBXC_DIR = directory for LibXC library
    CC = C-compiler with MPI
    CXX = C++-compiler with MPI
    FC = fortran-compiler
    LOADER = C++-compiler with MPI

ScalES can be compiled using `debug` or `release` mode by the variable `COMPILE_MODE` in `make.inc`. This variable mainly controls the compiling flag `-DRELEASE`. The `debug` mode introduces tracing of call stacks at all levels of functions, and may significantly slow down the code. For production runs, use `release` mode.

If `make.inc` is configured correctly,

    cd ${ScalES_DIR}
    cd src/common
    make
    cd ../pwdft
    make pwdft

should produce executable file `pwdft` under `pwdft/`.
