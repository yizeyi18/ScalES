set( DG_PWDFT_SRC
     eigensolver.cpp
     hamiltonian.cpp
     iondynamics.cpp
     pwdft.cpp
     scf.cpp
     spinor.cpp )

if( DGDFT_ENABLE_CUDA )

  set( DG_PWDFT_CUDA_SRC
       device_eigensolver.cpp
       device_hamiltonian.cpp
       device_spinor.cpp )

  list( TRANSFORM DG_PWDFT_CUDA_SRC PREPEND "device/" )
  list( APPEND DG_PWDFT_SRC ${DG_PWDFT_CUDA_SRC}      ) 

endif()

add_executable( pwdft ${DG_PWDFT_SRC} )
target_link_libraries( pwdft PUBLIC common )
