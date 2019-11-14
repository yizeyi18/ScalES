#include <cuda_wrapper.hpp>
#include <iostream>
#include <cassert>
  #include <exception>

#include <cublas_v2.h>

//#define CUDA_THROW_AS_ASSERT

#define CUDA_ASSERT(err)  { assert( err == cudaSuccess );           }
#define CUBLAS_ASSERT(err){ assert( err == CUBLAS_STATUS_SUCCESS ); }


#ifdef CUDA_THROW_AS_ASSERT
  #define CUDA_THROW(err) CUDA_ASSERT(err);
  #define CUBLAS_THROW(err) CUBLAS_ASSERT(err);
#else
  #define CUDA_THROW(err)  { if(err != cudaSuccess) throw cuda_exception( err );           }
  #define CUBLAS_THROW(err){ if(err != CUBLAS_STATUS_SUCCESS) throw cuda_exception( err ); }
#endif

namespace cuda {
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

class cuda_exception : public std::exception {

  std::string message;

  virtual const char* what() const throw() {
    return message.c_str();
  }

public:

  cuda_exception( const char* msg ) : std::exception(), message( msg ) { };
  cuda_exception( cudaError_t err ) : cuda_exception( cudaGetErrorString( err ) ) { } 
  cuda_exception( cublasStatus_t err ) : cuda_exception( cublasGetErrorString( err ) ) { }

};

namespace wrappers {

void memset( void* data, int val, size_t len ) {
  CUDA_THROW( cudaMemset( data, val, len ) );
}

void device_sync() {
  CUDA_THROW( cudaDeviceSynchronize() );
}

void memcpy_h2d( void* dest, const void* src, size_t len ) {
  CUDA_THROW( cudaMemcpy( dest, src, len, cudaMemcpyHostToDevice ) );
}

void memcpy_d2h( void* dest, const void* src, size_t len ) {
  CUDA_THROW( cudaMemcpy( dest, src, len, cudaMemcpyDeviceToHost ) );
}

void* malloc( size_t len ) {

  void* ptr;
  CUDA_THROW( cudaMalloc( &ptr, len ) );
  //std::cout << "CUDA MALLOC " << len << ", " << ptr << std::endl;
  return ptr;

}

void  free( void* ptr ) {
  //std::cout << "CUDA FREE " << ptr << std::endl;
  CUDA_THROW( cudaFree( ptr ) );
}

}


namespace detail {

  struct cuda_event_pimpl {

    cudaEvent_t event;

    cuda_event_pimpl(){
      CUDA_THROW( cudaEventCreate( &event ) );
    }

    ~cuda_event_pimpl() noexcept {
      CUDA_ASSERT( cudaEventDestroy( event ) );
    }

  };

  struct cuda_stream_pimpl {

    cudaStream_t stream;

    cuda_stream_pimpl(){
      CUDA_THROW( cudaStreamCreate( &stream ) );
    }

    ~cuda_stream_pimpl() noexcept {
      CUDA_ASSERT( cudaStreamDestroy( stream ) );
    }

  };

  struct cublas_handle_pimpl {

    cublasHandle_t handle;

    cublas_handle_pimpl(){
      CUBLAS_THROW( cublasCreate( &handle ) );
    }

    ~cublas_handle_pimpl() noexcept {
      CUBLAS_ASSERT( cublasDestroy( handle ) );
    }

  };

}


cuda_stream::cuda_stream() :
  pimpl_( std::make_unique<detail::cuda_stream_pimpl>() ){ }

cuda_stream::~cuda_stream() noexcept = default;
cuda_stream::cuda_stream( cuda_stream&& ) noexcept = default;

void cuda_stream::synchronize() const {
  CUDA_THROW( cudaStreamSynchronize( pimpl_->stream ) );
}






cuda_event::cuda_event() :
  pimpl_( std::make_unique<detail::cuda_event_pimpl>() ){ }

cuda_event::~cuda_event() noexcept = default;
cuda_event::cuda_event( cuda_event&& ) noexcept = default;

void cuda_event::record( const cuda_stream& stream ) {
  CUDA_THROW( cudaEventRecord( pimpl_->event, stream.pimpl_->stream ) );
}

void cuda_event::record() {
  CUDA_THROW( cudaEventRecord( pimpl_->event ) );
}

void cuda_event::synchronize() const {
  CUDA_THROW( cudaEventSynchronize( pimpl_->event ) );
}


float cuda_event::elapsed_time( const cuda_event& first, const cuda_event& second ) {
  float time;
  CUDA_THROW( cudaEventElapsedTime( &time, first.pimpl_->event, second.pimpl_->event ) );
  return time;
}

cublas_handle::cublas_handle() :
  pimpl_( std::make_unique<detail::cublas_handle_pimpl>() ){ }

cublas_handle::~cublas_handle() noexcept = default;
cublas_handle::cublas_handle( cublas_handle&& ) noexcept = default;

detail::cublas_handle_pimpl* cublas_handle::pimpl() const{
  return pimpl_.get();
};








template <>
void cublas_gemm_batched( cublas_handle& handle,
  char TRANSA, char TRANSB, int M, int N, int K, 
  double ALPHA, double** A_device, int LDA, double** B_device,
  int LDB, double BETA, double** C_device, int LDC, int batch_count ) {

  cublasOperation_t TA = cublasOpFromChar(TRANSA) ;
  cublasOperation_t TB = cublasOpFromChar(TRANSB) ;

  auto handle_h = handle.pimpl()->handle;

  CUBLAS_THROW(
    cublasDgemmBatched( handle_h, TA, TB, M, N, K, &ALPHA, A_device, LDA, B_device,
      LDB, &BETA, C_device, LDC, batch_count )
  )

}

template <>
void cublas_axpy( cublas_handle& handle,
  int N, double ALPHA, const double* X, int INCX, double* Y, int INCY 
) {

  auto handle_h = handle.pimpl()->handle;

  CUBLAS_THROW( cublasDaxpy( handle_h, N, &ALPHA, X, INCX, Y, INCY ) );

}

template <typename T>
__global__ void axpby_kernel( const size_t n, const T alpha, const T* X, const size_t INCX, const T beta, T* Y, const size_t INCY ) {

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid < n ) {
		const int tid_inc_y = tid * INCY;
		const int tid_inc_x = tid * INCX;
		Y[ tid_inc_y ] = beta * Y[ tid_inc_y ] + alpha * X[ tid_inc_x ];
	}
}


template <typename T>
void axpby_device( 
  int N, T ALPHA, const T* X, int INCX, T BETA, T* Y, int INCY
) {

  auto div = std::div( N, 1024 );
  
  axpby_kernel<T><<< div.quot + !!div.rem, 1024 >>>( 
    N, ALPHA, X, INCX, BETA, Y, INCY 
  );
  CUDA_THROW( cudaGetLastError() );
}

template
void axpby_device<double>( 
  int N, double ALPHA, const double* X, int INCX, double BETA, double* Y, int INCY
);

}
