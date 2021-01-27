#
#   This file is a part of ScalES (see LICENSE). All Right Reserved
#
#   Copyright (c) 2012-2021 The Regents of the University of California,
#   through Lawrence Berkeley National Laboratory.  
#
#   Authors: David Williams-Young
#

include( scales-linalg )
include( scales-libxc  )


find_package( FFTW3 REQUIRED COMPONENTS MPI )


add_library( ScalES::tpl_depends INTERFACE IMPORTED )
target_link_libraries( ScalES::tpl_depends INTERFACE Libxc::xc        )
target_link_libraries( ScalES::tpl_depends INTERFACE FFTW3::FFTW3_MPI )
target_link_libraries( ScalES::tpl_depends INTERFACE ScalES::linalg    )

# PEXSI
if( ScalES_ENABLE_PEXSI )

  # Find and link to PEXSI, handles SuperLU/ParMETIS/SymPACK
  find_package( PEXSI REQUIRED )

  add_library( ScalES::PEXSI INTERFACE IMPORTED )
  target_link_libraries( ScalES::PEXSI INTERFACE PEXSI::PEXSI )
  target_compile_definitions( ScalES::PEXSI INTERFACE "-D PEXSI" )

  target_link_libraries( ScalES::tpl_depends INTERFACE ScalES::PEXSI )

endif()
