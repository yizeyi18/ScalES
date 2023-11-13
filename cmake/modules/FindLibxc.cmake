#   FindLibxc.cmake
#
#   Finds the Libxc library
#
#   This module will define the following variables:
#
#     Libxc_FOUND         - System has found Libxc installation
#     Libxc_INCLUDE_DIR   - Location of Libxc headers
#     Libxc_LIBRARIES     - Libxc libraries
#
#   This module can handle the following COMPONENTS
#
#     MPI     - MPI version
#
#    This module will export the following targets if Libxc_FOUND
#
#      Libxc::Libxc
#
#    This module will export the following targets if 
#    Libxc_MPI_FOUND
#
#      Libxc::Libxc_MPI
#
#
#    Proper usage:
#
#      project( TEST_FIND_Libxc C )
#      find_package( Libxc )
#
#      if( Libxc_FOUND )
#        add_executable( test test.cxx )
#        target_link_libraries( test Libxc::Libxc )
#      endif()
#
#   This module will use the following variables to change
#   default behaviour if set
#
#     Libxc_PREFIX
#     Libxc_INCLUDE_DIR
#     Libxc_LIBRARY_DIR
#     Libxc_LIBRARIES
#
#==================================================================
#   Copyright (c) 2018 The Regents of the University of California,
#   through Lawrence Berkeley National Laboratory.  
#
#   Author: David Williams-Young
#   
#   This file is part of cmake-modules. All rights reserved.
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
#==================================================================

cmake_minimum_required( VERSION 3.11 ) # Require CMake 3.11+
include(FindPackageHandleStandardArgs)


# Set up some auxillary vars if hints have been set

if( Libxc_PREFIX AND NOT Libxc_LIBRARY_DIR )
  set( Libxc_LIBRARY_DIR 
    ${Libxc_PREFIX}/lib 
  )
endif()







# Try to find the header
find_path( Libxc_INCLUDE_DIR 
  NAMES xc.h
  HINTS ${Libxc_PREFIX}
  PATH_SUFFIXES include
  DOC "Location of Libxc header"
)



# Try to serial find libraries if not already set
if( NOT Libxc_LIBRARIES )

  find_library( Libxc_LIBRARIES
    NAMES xc 
    HINTS ${Libxc_PREFIX}
    PATHS ${Libxc_LIBRARY_DIR}
    PATH_SUFFIXES lib
    DOC "Libxc Library"
  )

endif()


# TODO: Add linkage testing

# fftw3-MPI
#if( "MPI" IN_LIST Libxc_FIND_COMPONENTS )


  # Try to find the header
#  find_path( Libxc_MPI_INCLUDE_DIR 
#    NAMES fftw3-mpi.h
#    HINTS ${Libxc_PREFIX} ${Libxc_MPI_PREFIX}
#    PATH_SUFFIXES include
#    DOC "Location of Libxc-MPI header"
#  )
  
  
  
  # Try to serial find libraries if not already set
#  if( NOT fftw3_MPI_LIBRARIES )
  
#    find_library( Libxc_MPI_LIBRARIES
#      NAMES fftw3_mpi 
#      HINTS ${Libxc_PREFIX} ${Libxc_MPI_PREFIX}
#      PATHS ${Libxc_LIBRARY_DIR} ${Libxc_MPI_LIBRARY_DIR}
#      PATH_SUFFIXES lib lib64 lib32
#      DOC "Libxc of Libxc-MPI Library"
#    )
  
#  endif()
  

#  if( Libxc_MPI_INCLUDE_DIR AND Libxc_MPI_LIBRARIES )
#    set( Libxc_MPI_FOUND TRUE )
#  endif()

  # MPI
#  if( NOT TARGET MPI::MPI_C )
#    include(CMakeFindDependencyMacro)
#    find_dependency( MPI REQUIRED )
#  endif()
#
#endif()


mark_as_advanced( Libxc_FOUND Libxc_INCLUDE_DIR
                  Libxc_LIBRARIES )

find_package_handle_standard_args( Libxc
  REQUIRED_VARS Libxc_INCLUDE_DIR Libxc_LIBRARIES
  HANDLE_COMPONENTS 
)


if( Libxc_FOUND AND NOT TARGET Libxc::xc )

  add_library( Libxc::xc INTERFACE IMPORTED )
  set_target_properties( Libxc::xc PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Libxc_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "${Libxc_LIBRARIES}"
  )

endif()


#if( Libxc_MPI_FOUND AND NOT TARGET Libxc::Libxc_MPI )
#
#  add_library( Libxc::Libxc_MPI INTERFACE IMPORTED )
#  set_target_properties( Libxc::Libxc_MPI PROPERTIES
#    INTERFACE_INCLUDE_DIRECTORIES "${Libxc_MPI_INCLUDE_DIR}"
#    INTERFACE_LINK_LIBRARIES      "${Libxc_MPI_LIBRARIES};Libxc::Libxc;MPI::MPI_C"
#  )
#
#endif()


