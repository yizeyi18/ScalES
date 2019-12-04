#   Copyright (c) 2018 The Regents of the University of California,
#   through Lawrence Berkeley National Laboratory.  
#
#   Author: David Williams-Young
#   
#   This file is part of PEXSI. All rights reserved.
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

set( LIBDG_SRC scf_dg.cpp scf.cpp eigensolver.cpp iondynamics.cpp 
	       hamiltonian_dg_matrix.cpp hamiltonian_dg_matrix_device.cpp hamiltonian_dg.cpp 
               hamiltonian.cpp periodtable.cpp spinor.cpp 
               fourier.cpp utility.cpp esdf.cpp mpi_interf.cpp 
               scalapack.cpp lapack.cpp blas.cpp global.cpp 
               tddft.cpp sgmres.cpp 
#cuda_memory_resource.cxx segregated_memory_resource.cxx 
cuda_scf_dg.cxx 
)


add_library( libdgdft ${LIBDG_SRC} )

if( DGDFT_ENABLE_CUDA )

  find_package( CUDA REQUIRED )
  message(STATUS "CUBLAS ${CUDA_CUBLAS_LIBRARIES}")
  set( LIBDG_DEVICE_SRC cuda_api_wrappers.cu cuda_type_wrappers.cu cublas_wrappers.cu axpby.cu )
  add_library( libdgdft_device ${LIBDG_DEVICE_SRC} )
  target_link_libraries( libdgdft_device
    PUBLIC
      DGDFT::compile_definitions
      DGDFT::parallel_cxx
      DGDFT::parallel_c
      DGDFT::external
      DGDFT::tpl_depends
      ${CUDA_CUBLAS_LIBRARIES}
  )

target_compile_features( libdgdft_device PRIVATE cxx_std_14 )
  target_link_libraries(libdgdft PUBLIC libdgdft_device )

  target_include_directories( libdgdft_device 
    PUBLIC
      $<INSTALL_INTERFACE:include>
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cuda_include>
  )
endif()

# TARGET properties

set_target_properties( libdgdft PROPERTIES PREFIX      ""         )
set_target_properties( libdgdft PROPERTIES OUTPUT_NAME "libdgdft" )

target_link_libraries( libdgdft
  PUBLIC
    DGDFT::compile_definitions
    DGDFT::parallel_cxx
    DGDFT::parallel_c
    DGDFT::external
    DGDFT::tpl_depends
)
target_compile_features( libdgdft PUBLIC cxx_std_17 )

target_include_directories( libdgdft 
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
