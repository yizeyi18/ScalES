#include <cublas_wrappers.hpp>
#include <exceptions.hpp>
#include <cublas_v2.h>

#include "cuda_type_pimpl.hpp"

const char* cublasGetErrorString( cublasStatus_t status ) {

  if( status == CUBLAS_STATUS_SUCCESS )
    return "SUCCESS";
  else if( status == CUBLAS_STATUS_NOT_INITIALIZED )
    return "NOT INITIALIZED";
  else if( status == CUBLAS_STATUS_ALLOC_FAILED )
    return "ALLOC FAILED";
  else if( status == CUBLAS_STATUS_INVALID_VALUE )
    return "INVALID VALUE";
  else if( status == CUBLAS_STATUS_ARCH_MISMATCH )
    return "ARCH MISMATCH";
  else if( status == CUBLAS_STATUS_MAPPING_ERROR )
    return "MAPPING ERROR";
  else if( status == CUBLAS_STATUS_EXECUTION_FAILED )
    return "EXECUTION FAILED";
  else if( status == CUBLAS_STATUS_INTERNAL_ERROR )
    return "INTERNAL ERROR";
  else if( status == CUBLAS_STATUS_NOT_SUPPORTED )
    return "NOT SUPPORTED";
  else if( status == CUBLAS_STATUS_LICENSE_ERROR )
    return "INVALID LICENSE";
  else 
    return "CUBLAS ERROR NOT RECOGNIZED";
  

}

cublasOperation_t cublasOpFromChar(char op){
	switch (op) {
		case 'n':
		case 'N':
			return CUBLAS_OP_N;
		case 't':
		case 'T':
			return CUBLAS_OP_T;
		case 'c':
		case 'C':
			return CUBLAS_OP_C;
    default:
      printf("UNKNOWN CUBLAS OP - DEFAULTING TO CUBLAS_OP_N");
      return CUBLAS_OP_N;
	}
}






namespace cublas {
namespace detail {

  struct handle_pimpl {

    cublasHandle_t handle;

    handle_pimpl(){
      CUBLAS_THROW( cublasCreate( &handle ) );
    }

    ~handle_pimpl() noexcept {
      CUBLAS_ASSERT( cublasDestroy( handle ) );
    }

  };

  handle_pimpl* get_handle( handle& h ) {
    return h.pimpl_.get();
  }

}



handle::handle() :
  pimpl_( std::make_shared<detail::handle_pimpl>() ){ }

handle::~handle() noexcept = default;
handle::handle( handle&& ) noexcept = default;
handle::handle( const handle& ) = default;

cuda::stream handle::get_stream() const {
  cudaStream_t stream;
  CUBLAS_THROW( cublasGetStream( pimpl_->handle, &stream ) );
  cuda::stream return_stream( std::make_shared<cuda::detail::stream_pimpl>() );
  return_stream.pimpl_->stream = stream;
  return return_stream;
}

void handle::set_stream( const cuda::stream& stream ) {
  CUBLAS_THROW( cublasSetStream( pimpl_->handle, 
                                 stream.pimpl_->stream ) );
}

namespace blas {

template <>
void gemm_batched( handle& handle,
  char TRANSA, char TRANSB, int M, int N, int K, 
  double ALPHA, double** A_device, int LDA, double** B_device,
  int LDB, double BETA, double** C_device, int LDC, int batch_count ) {

  cublasOperation_t TA = cublasOpFromChar(TRANSA) ;
  cublasOperation_t TB = cublasOpFromChar(TRANSB) ;

  auto handle_h = detail::get_handle(handle)->handle;

  CUBLAS_THROW(
    cublasDgemmBatched( handle_h, TA, TB, M, N, K, &ALPHA, A_device, LDA, B_device,
      LDB, &BETA, C_device, LDC, batch_count )
  )

}

template <>
void axpy( handle& handle,
  int N, double ALPHA, const double* X, int INCX, double* Y, int INCY 
) {

  auto handle_h = detail::get_handle(handle)->handle;

  CUBLAS_THROW( cublasDaxpy( handle_h, N, &ALPHA, X, INCX, Y, INCY ) );

}

}

}
