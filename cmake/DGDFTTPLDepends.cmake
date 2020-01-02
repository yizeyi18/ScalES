#   Copyright (c) 2018 The Regents of the University of California,
#   through Lawrence Berkeley National Laboratory.  
#
#   Author: David Williams-Young
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

# BLAS / LAPACK
#find_package( BLAS   REQUIRED OPTIONAL_COMPONENTS lp64 )
find_package( LAPACK REQUIRED OPTIONAL_COMPONENTS lp64 blacs scalapack )

add_library( DGDFT::linalg INTERFACE IMPORTED )
target_link_libraries( DGDFT::linalg INTERFACE LAPACK::lapack )
if( BLAS_USES_UNDERSCORE )
  target_compile_definitions( DGDFT::linalg INTERFACE "-D Add_" )
endif()
  




find_package( ScaLAPACK REQUIRED                )
find_package( Libxc     REQUIRED                )
find_package( FFTW3     REQUIRED COMPONENTS MPI )


add_library( DGDFT::tpl_depends INTERFACE IMPORTED )
target_link_libraries( DGDFT::tpl_depends INTERFACE ScaLAPACK::scalapack )
target_link_libraries( DGDFT::tpl_depends INTERFACE Libxc::xc            )
target_link_libraries( DGDFT::tpl_depends INTERFACE FFTW3::fftw3_mpi     )
target_link_libraries( DGDFT::tpl_depends INTERFACE DGDFT::linalg        )

# PEXSI
if( DGDFT_ENABLE_PEXSI )

  # Find and link to PEXSI, handles SuperLU/ParMETIS/SymPACK
  find_package( PEXSI REQUIRED )

  add_library( DGDFT::PEXSI INTERFACE IMPORTED )
  target_link_libraries( DGDFT::PEXSI INTERFACE PEXSI::PEXSI )
  target_compile_definitions( DGDFT::PEXSI INTERFACE "-D PEXSI" )

  target_link_libraries( DGDFT::tpl_depends INTERFACE DGDFT::PEXSI         )
endif()
