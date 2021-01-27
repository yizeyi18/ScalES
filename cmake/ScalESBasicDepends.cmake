#
#   This file is a part of ScalES (see LICENSE). All Right Reserved
#
#   Copyright (c) 2012-2021 The Regents of the University of California,
#   through Lawrence Berkeley National Laboratory.  
#
#   Authors: David Williams-Young
#


# Handle MPI 
find_package( MPI REQUIRED )

add_library( ScalES::parallel_cxx     INTERFACE IMPORTED )
add_library( ScalES::parallel_c       INTERFACE IMPORTED )
add_library( ScalES::parallel_fortran INTERFACE IMPORTED )

target_link_libraries( ScalES::parallel_cxx     INTERFACE MPI::MPI_CXX     )
target_link_libraries( ScalES::parallel_c       INTERFACE MPI::MPI_C       )
target_link_libraries( ScalES::parallel_fortran INTERFACE MPI::MPI_Fortran )

# Handle OpenMP
if( ScalES_ENABLE_OPENMP )

  find_package( OpenMP REQUIRED )

  target_link_libraries( ScalES::parallel_cxx     INTERFACE OpenMP::OpenMP_CXX     )
  target_link_libraries( ScalES::parallel_c       INTERFACE OpenMP::OpenMP_C       )
  target_link_libraries( ScalES::parallel_fortran INTERFACE OpenMP::OpenMP_Fortran )

endif()

# Handle CUDA Toolkit
if( ScalES_ENABLE_CUDA )

  find_package( CUDAToolkit REQUIRED )

  add_library( ScalES::cuda INTERFACE IMPORTED )
  target_link_libraries( ScalES::cuda INTERFACE CUDA::cublas CUDA::cufft CUDA::cusolver )
  
endif()
