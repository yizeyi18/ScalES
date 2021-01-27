#
#   This file is a part of ScalES (see LICENSE). All Right Reserved
#
#   Copyright (c) 2012-2021 The Regents of the University of California,
#   through Lawrence Berkeley National Laboratory.  
#
#   Authors: David Williams-Young
#
find_package( Libxc QUIET )
if( NOT Libxc_FOUND ) 

  message( STATUS "Libxc Not Found... Building!" )

  include(FetchContent)
  FetchContent_Declare(
    libxc
    GIT_REPOSITORY https://gitlab.com/libxc/libxc.git
    GIT_TAG        5.0.0
  )

  
  set( BUILD_TESTING OFF CACHE BOOL "" )
  FetchContent_MakeAvailable( libxc )
  add_library( Libxc::xc ALIAS xc )
  target_include_directories( xc 
    PUBLIC 
      $<BUILD_INTERFACE:${libxc_SOURCE_DIR}/src>
      $<BUILD_INTERFACE:${libxc_BINARY_DIR}/src>
      $<BUILD_INTERFACE:${libxc_BINARY_DIR}>
      $<BUILD_INTERFACE:${libxc_BINARY_DIR}/gen_funcidx>
  )
  
else()
  message( STATUS "Libxc Found" )
  message( STATUS "Libxc_LIBRARIES = ${Libxc_LIBRARIES}" )
endif()

