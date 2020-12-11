
set( DG_COMMON_SRC
     blas.cpp  
     esdf.cpp 
     fourier.cpp  
     global.cpp  
     lapack.cpp  
     mpi_interf.cpp  
     periodtable.cpp  
     scalapack.cpp  
     utility.cpp )

if( DGDFT_ENABLE_CUDA )

  set( DG_COMMON_CUDA_SRC
       device_utility.cpp
       device_blas.cpp
       device_error.cpp
       device_fft.cpp
       device_mpi_interf.cpp
       device_solver.cpp
       device_utilis.cu )

  list( TRANSFORM DG_COMMON_CUDA_SRC PREPEND "device/" )
  list( APPEND DG_COMMON_SRC ${DG_COMMON_CUDA_SRC} )

endif()

add_library( common ${DG_COMMON_SRC} )
target_include_directories( common PUBLIC ${CMAKE_CURRENT_LIST_DIR} )
target_link_libraries( common PUBLIC 
  DGDFT::compile_definitions
  DGDFT::parallel_c
  DGDFT::external
  DGDFT::tpl_depends
)

if( TARGET DGDFT::cuda )
  target_link_libraries( common DGDFT::cuda )
endif()
