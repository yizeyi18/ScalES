#
#   This file is a part of ScalES (see LICENSE). All Right Reserved
#
#   Copyright (c) 2012-2021 The Regents of the University of California,
#   through Lawrence Berkeley National Laboratory.  
#
#   Authors: David Williams-Young
#




add_library( ScalES::compile_definitions INTERFACE IMPORTED )

# C++14
#target_compile_features( ScalES::compile_definitions INTERFACE cxx_std_14 )

# Performance Profiling
if( ScalES_ENABLE_PROFILE )
  target_compile_options( ScalES::compile_definitions INTERFACE -g -pg )
endif( ScalES_ENABLE_PROFILE )


# Handle DEBUG / RELEASE flags
if( CMAKE_BUILD_TYPE MATCHES Release )

  target_compile_definitions( ScalES::compile_definitions INTERFACE "-D RELEASE" )

else()

  if( NOT ScalES_DEBUG_LEVEL )
    set( ScalES_DEBUG_LEVEL 1 )
  endif( NOT ScalES_DEBUG_LEVEL )

  target_compile_definitions( ScalES::compile_definitions INTERFACE "-D DEBUG=${ScalES_DEBUG_LEVEL}" )

endif()



# COMPLEX flags
if( ScalES_ENABLE_COMPLEX )
  target_compile_definitions( ScalES::compile_definitions INTERFACE "-D CPX" )
endif()

# DEVICE flags
if( ScalES_ENABLE_DEVICE )
  target_compile_definitions( ScalES::compile_definitions INTERFACE "-D DEVICE" )
endif()
if( ScalES_ENABLE_GPUDIRECT )
  target_compile_definitions( ScalES::compile_definitions INTERFACE "-D GPUDIRECT" )
endif()

