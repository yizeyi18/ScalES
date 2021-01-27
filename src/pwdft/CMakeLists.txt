#
#   This file is a part of ScalES (see LICENSE). All Right Reserved
#
#   Copyright (c) 2012-2021 The Regents of the University of California,
#   through Lawrence Berkeley National Laboratory.  
#
#   Authors: David Williams-Young
#
set( DG_PWDFT_SRC
     eigensolver.cpp
     hamiltonian.cpp
     iondynamics.cpp
     pwdft.cpp
     scf.cpp
     spinor.cpp )

if( ScalES_ENABLE_CUDA )

  set( DG_PWDFT_CUDA_SRC
       device_eigensolver.cpp
       device_hamiltonian.cpp
       device_spinor.cpp )

  list( TRANSFORM DG_PWDFT_CUDA_SRC PREPEND "device/" )
  list( APPEND DG_PWDFT_SRC ${DG_PWDFT_CUDA_SRC}      ) 

endif()

add_executable( pwdft ${DG_PWDFT_SRC} )
target_link_libraries( pwdft PUBLIC common )
target_include_directories( pwdft PUBLIC ${PROJECT_SOURCE_DIR}/src/pwdft )
#target_compile_options( pwdft PUBLIC "-Wall" )
