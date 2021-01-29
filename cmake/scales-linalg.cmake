#
#   This file is a part of ScalES (see LICENSE). All Right Reserved
#
#   Copyright (c) 2012-2021 The Regents of the University of California,
#   through Lawrence Berkeley National Laboratory.  
#
#   Authors: David Williams-Young
#

include( FetchContent )

# ScaLAPACK / BLAS / LAPACK
find_package( ScaLAPACK REQUIRED )


# ScaLAPACK++ / BLACS++
FetchContent_Declare( scalapackpp
  GIT_REPOSITORY https://github.com/wavefunction91/scalapackpp.git
  GIT_TAG        feature/ilp64
)
FetchContent_MakeAvailable( scalapackpp )

# BLAS++/LAPACK++
FetchContent_Declare( blaspp
  GIT_REPOSITORY https://bitbucket.org/icl/blaspp.git 
)
FetchContent_Declare( lapackpp
  GIT_REPOSITORY https://bitbucket.org/icl/lapackpp.git 
)


set( use_openmp ${ScaES_ENABLE_OPENMP} CACHE BOOL "BLAS++/LAPACK++ OpenMP Bindings" )
FetchContent_MakeAvailable( blaspp )
FetchContent_MakeAvailable( lapackpp )
target_compile_definitions( lapackpp PUBLIC LAPACK_COMPLEX_CPP )



add_library( ScalES::linalg INTERFACE IMPORTED )
target_link_libraries( ScalES::linalg INTERFACE scalapackpp::scalapackpp lapackpp ) 
