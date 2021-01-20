# BLAS / LAPACK
find_package( ScaLAPACK REQUIRED )

add_library( DGDFT::linalg INTERFACE IMPORTED )
#target_link_libraries( DGDFT::linalg INTERFACE ScaLAPACK::ScaLAPACK )
#if( BLAS_FORTRAN_UNDERSCORE )
#  target_compile_definitions( DGDFT::linalg INTERFACE "-D Add_" )
#endif()

# BLAS++/LAPACK++
include( FetchContent )
FetchContent_Declare( blaspp
  GIT_REPOSITORY https://bitbucket.org/icl/blaspp.git 
)
FetchContent_Declare( lapackpp
  GIT_REPOSITORY https://bitbucket.org/icl/lapackpp.git 
)

FetchContent_MakeAvailable( blaspp )
FetchContent_MakeAvailable( lapackpp )
target_compile_definitions( lapackpp PUBLIC LAPACK_COMPLEX_CPP )

target_link_libraries( DGDFT::linalg INTERFACE lapackpp ScaLAPACK::ScaLAPACK ) 
