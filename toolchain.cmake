set( CMAKE_C_COMPILER gcc )
set( CMAKE_CXX_COMPILER g++ )
set( CMAKE_Fortran_COMPILER gfortran )

#set( MPI_C_COMPILER mpicc )
#set( MPI_CXX_COMPILER mpicxx )
#set( MPI_Fortran_COMPILER mpifort )
#set( MPI_cuda_COMPILER mpicxx )


set( INTEL_ROOT "/opt/intel/compilers_and_libraries_2019.3.199" )
set( intelmkl_PREFIX ${INTEL_ROOT}/linux/mkl )

#set( fftw3_PREFIX "/opt/cray/pe/fftw/3.3.8.3/x86_64/" )
set( fftw3_PREFIX "/usr/common/software/fftw3/3.3.8/gcc/8.2.0/skx" )

