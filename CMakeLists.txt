#   Copyright (c) 2020 The Regents of the University of California,
#   through Lawrence Berkeley National Laboratory.  
#   
#   This file is part of DGDFT. All rights reserved.
#   
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#   
#   (1) Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#   (2) Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#   (3) Neither the name of the University of California, Lawrence Berkeley
#   National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#   
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
#   ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#   
#   You are under no obligation whatsoever to provide any bug fixes, patches, or
#   upgrades to the features, functionality or performance of the source code
#   ("Enhancements") to anyone; however, if you choose to make your Enhancements
#   available either publicly, or directly to Lawrence Berkeley National
#   Laboratory, without imposing a separate written license agreement for such
#   Enhancements, then you hereby grant the following license: a non-exclusive,
#   royalty-free perpetual license to install, use, modify, prepare derivative
#   works, incorporate into other computer software, distribute, and sublicense
#   such enhancements or derivative works thereof, in binary and source code form.
#
cmake_minimum_required( VERSION 3.17 ) # Require CMake 3.17+

# Set up project definition + version information
# TODO: Get version number
project( DGDFT VERSION 0.0.0 LANGUAGES CXX C Fortran )


# General DGDFT Options
option( DGDFT_ENABLE_COMPLEX   "Enable Complex Arithmetic"    OFF )
option( DGDFT_ENABLE_OPENMP    "Enable OpenMP Parallelism"    ON  )
option( DGDFT_ENABLE_PROFILE   "Enable Performance Profiling" OFF )
option( DGDFT_ENABLE_PEXSI     "Enable PEXSI Bindings"        OFF )

# GPU Related DGDFT Options
option( DGDFT_ENABLE_CUDA  "Enable CUDA Bindings"  OFF )
option( DGDFT_ENABLE_MAGMA "Enable MAGMA Bingings" OFF )

# If CUDA enabled, enable device bindings
if( DGDFT_ENABLE_CUDA )
  set( DGDFT_ENABLE_DEVICE ON CACHE BOOL "Enable Device Bindings" )
endif()


# If CUDA enabled, enable CUDA as a language
if( DGDFT_ENABLE_CUDA )
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
set( GH_TOKEN 773b091d7a41133e2558c10f4999450fc2a09450 )
FetchContent_Declare( linalg-cmake-modules 
  GIT_REPOSITORY https://${GH_TOKEN}@github.com/wavefunction91/linalg-cmake-modules.git
  GIT_TAG        main
)
FetchContent_GetProperties( linalg-cmake-modules )
if( NOT linalg-cmake-modules_POPULATED )
  FetchContent_Populate( linalg-cmake-modules )
  list( APPEND CMAKE_MODULE_PATH ${linalg-cmake-modules_SOURCE_DIR} )
endif()




# Setup Dependencies / Flags
include( DGDFTCompileFlags ) # Compile Flags
include( DGDFTBasicDepends ) # Basic Dependencies
include( DGDFTTPLDepends   ) # TPL Dependencies


# External Utilities
add_subdirectory( external )

# Build libdgdft etc
add_subdirectory( src )

# Build pw/dg executables
#add_subdirectory( examples )

