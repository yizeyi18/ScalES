
set( DG_COMMON_SRC
     blas.cpp  
     esdf.cpp 
     fourier.cpp  
     global.cpp  
     lapack.cpp  
     mpi_interf.cpp  
     periodtable.cpp  
     scalapack.cpp  
     utility.cpp 
     block_distributor_impl.cpp
     host_block_distributor.cpp
)

if( DGDFT_ENABLE_CUDA )

  set( DG_COMMON_CUDA_SRC
       device/device_utility.cpp
       device/nvgpu/device_blas.cpp
       device/nvgpu/device_error.cpp
       device/nvgpu/device_fft.cpp
       device/nvgpu/device_mpi_interf.cpp
       device/nvgpu/device_solver.cpp
       device/nvgpu/device_utils.cu )

  #list( TRANSFORM DG_COMMON_CUDA_SRC PREPEND "device/nvgpu/" )
  list( APPEND DG_COMMON_SRC ${DG_COMMON_CUDA_SRC} )

endif()

add_library( common ${DG_COMMON_SRC} )
target_include_directories( common PUBLIC 
  ${PROJECT_SOURCE_DIR}/src/common
  ${PROJECT_BINARY_DIR}/src/common
)
target_link_libraries( common PUBLIC 
  DGDFT::compile_definitions
  DGDFT::parallel_c
  DGDFT::external
  DGDFT::tpl_depends
)

if( DGDFT_ENABLE_CUDA )
  target_link_libraries( common PUBLIC DGDFT::cuda )
  target_include_directories( common PUBLIC 
    ${PROJECT_SOURCE_DIR}/src/common/device
    ${PROJECT_SOURCE_DIR}/src/common/device/nvgpu
  )
endif()



# Set up config file
set( DG_PP_ONCV_PATH ${PROJECT_SOURCE_DIR}/examples/pp_ONCV )
configure_file(
  ${PROJECT_SOURCE_DIR}/src/common/config.hpp.in
  ${PROJECT_BINARY_DIR}/src/common/config.hpp
)
target_compile_definitions( common PUBLIC DG_HAS_CONFIG )
