#
#   This file is a part of ScalES (see LICENSE). All Right Reserved
#
#   Copyright (c) 2012-2021 The Regents of the University of California,
#   through Lawrence Berkeley National Laboratory.  
#
#   Authors: David Williams-Young
#

# BLAS / LAPACK
find_package( ScaLAPACK REQUIRED )

add_library( ScalES::linalg INTERFACE IMPORTED )
#target_link_libraries( ScalES::linalg INTERFACE ScaLAPACK::ScaLAPACK )
#if( BLAS_FORTRAN_UNDERSCORE )
#  target_compile_definitions( ScalES::linalg INTERFACE "-D Add_" )
#endif()

# BLAS++/LAPACK++
#include( FetchContent )
#FetchContent_Declare( blaspp
#  GIT_REPOSITORY https://bitbucket.org/icl/blaspp.git 
#)
#FetchContent_Declare( lapackpp
#  GIT_REPOSITORY https://bitbucket.org/icl/lapackpp.git 
#)

#FetchContent_MakeAvailable( blaspp )
#FetchContent_MakeAvailable( lapackpp )
find_package( lapackpp REQUIRED )
#target_compile_definitions( lapackpp PUBLIC LAPACK_COMPLEX_CPP )

target_link_libraries( ScalES::linalg INTERFACE lapackpp ScaLAPACK::ScaLAPACK ) 
