#
#   This file is a part of ScalES (see LICENSE). All Right Reserved
#
#   Copyright (c) 2012-2021 The Regents of the University of California,
#   through Lawrence Berkeley National Laboratory.  
#
#   Authors: David Williams-Young
#
cmake_minimum_required( VERSION 3.17 ) # Require CMake 3.17+

# Set up project definition + version information
# TODO: Get version number
project( ScalES VERSION 0.0.0 LANGUAGES CXX C Fortran )


# General ScalES Options
option( ScalES_ENABLE_COMPLEX   "Enable Complex Arithmetic"    OFF )
option( ScalES_ENABLE_OPENMP    "Enable OpenMP Parallelism"    ON  )
option( ScalES_ENABLE_PROFILE   "Enable Performance Profiling" OFF )
option( ScalES_ENABLE_PEXSI     "Enable PEXSI Bindings"        OFF )

# GPU Related ScalES Options
option( ScalES_ENABLE_CUDA  "Enable CUDA Bindings"  OFF )
option( ScalES_ENABLE_MAGMA "Enable MAGMA Bingings" OFF )

# If CUDA enabled, enable device bindings
if( ScalES_ENABLE_CUDA )
  set( ScalES_ENABLE_DEVICE ON CACHE BOOL "Enable Device Bindings" )
endif()


# If CUDA enabled, enable CUDA as a language
if( ScalES_ENABLE_CUDA )
  enable_language( CUDA )
endif()


# Append local cmake directory to find CMAKE Modules
if( CMAKE_MODULE_PATH )
  list( APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
else()
  set( CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
endif()
list( APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

# Get Modifed BLAS/LAPACK/ScaLAPACK Modules
include( FetchContent )
FetchContent_Declare( linalg-cmake-modules 
  GIT_REPOSITORY https://github.com/wavefunction91/linalg-cmake-modules.git
  GIT_TAG        main
)
FetchContent_GetProperties( linalg-cmake-modules )
if( NOT linalg-cmake-modules_POPULATED )
  FetchContent_Populate( linalg-cmake-modules )
  list( APPEND CMAKE_MODULE_PATH ${linalg-cmake-modules_SOURCE_DIR} )
endif()




# Setup Dependencies / Flags
include( ScalESCompileFlags ) # Compile Flags
include( ScalESBasicDepends ) # Basic Dependencies
include( ScalESTPLDepends   ) # TPL Dependencies


# External Utilities
add_subdirectory( external )

# Build libscales etc
add_subdirectory( src )

# Build pw/dg executables
#add_subdirectory( examples )

