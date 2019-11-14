#include <cuda_wrapper.hpp>
#include <cassert>
#include <iostream>

#include <cublas_v2.h>

namespace cuda {
namespace wrappers {

void memset( void* data, int val, size_t len ) {
  auto status = cudaMemset( data, val, len );
  assert( status == cudaSuccess );
}

void device_sync() {
  auto status = cudaDeviceSynchronize();
  assert( status == cudaSuccess );
}

void memcpy_h2d( void* dest, const void* src, size_t len ) {
  auto status = cudaMemcpy( dest, src, len, cudaMemcpyHostToDevice );
  assert( status == cudaSuccess );
}

void memcpy_d2h( void* dest, const void* src, size_t len ) {
  auto status = cudaMemcpy( dest, src, len, cudaMemcpyDeviceToHost );
  assert( status == cudaSuccess );
}

void* malloc( size_t len ) {

  void* ptr;
  auto status =cudaMalloc( &ptr, len );
  assert( status == cudaSuccess );
  std::cout << "CUDA MALLOC " << len << ", " << ptr << std::endl;
  return ptr;

}

void  free( void* ptr ) {
  std::cout << "CUDA FREE " << ptr << std::endl;
  cudaFree( ptr );
}

}


namespace detail {

  struct cuda_event_pimpl {

    cudaEvent_t event;

    cuda_event_pimpl(){
      auto status = cudaEventCreate( &event );
      assert( status == cudaSuccess );
    }

    ~cuda_event_pimpl() noexcept {
      auto status = cudaEventDestroy( event );
      assert( status == cudaSuccess );
    }

  };

  struct cuda_stream_pimpl {

    cudaStream_t stream;

    cuda_stream_pimpl(){
      auto status = cudaStreamCreate( &stream );
      assert( status == cudaSuccess );
    }

    ~cuda_stream_pimpl() noexcept {
      auto status = cudaStreamDestroy( stream );
      assert( status == cudaSuccess );
    }

  };

  struct cublas_handle_pimpl {

    cublasHandle_t handle;

    cublas_handle_pimpl(){
      auto status = cublasCreate( &handle );
      assert( status == CUBLAS_STATUS_SUCCESS );
    }

    ~cublas_handle_pimpl() noexcept {
      auto status = cublasDestroy( handle );
      assert( status == CUBLAS_STATUS_SUCCESS );
    }

  };

}


cuda_stream::cuda_stream() :
  pimpl_( std::make_unique<detail::cuda_stream_pimpl>() ){ }

cuda_stream::~cuda_stream() noexcept = default;
cuda_stream::cuda_stream( cuda_stream&& ) noexcept = default;

void cuda_stream::synchronize() const {
  auto status = cudaStreamSynchronize( pimpl_->stream );
  assert( status == cudaSuccess );
}






cuda_event::cuda_event() :
  pimpl_( std::make_unique<detail::cuda_event_pimpl>() ){ }

cuda_event::~cuda_event() noexcept = default;
cuda_event::cuda_event( cuda_event&& ) noexcept = default;

void cuda_event::record( const cuda_stream& stream ) {
  auto status = cudaEventRecord( pimpl_->event, stream.pimpl_->stream );
  assert( status == cudaSuccess );
}

void cuda_event::record() {
  auto status = cudaEventRecord( pimpl_->event );
  assert( status == cudaSuccess );
}

void cuda_event::synchronize() const {
  auto status = cudaEventSynchronize( pimpl_->event );
  assert( status == cudaSuccess );
}


float cuda_event::elapsed_time( const cuda_event& first, const cuda_event& second ) {
  float time;
  auto status = cudaEventElapsedTime( &time, 
    first.pimpl_->event, second.pimpl_->event );
  assert( status == cudaSuccess );
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

  cublasOperation_t TA = TRANSA == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t TB = TRANSB == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;

  auto handle_h = handle.pimpl()->handle;

  auto stat = 
  cublasDgemmBatched( handle_h, TA, TB, M, N, K, &ALPHA, A_device, LDA, B_device,
    LDB, &BETA, C_device, LDC, batch_count );

  assert( stat == CUBLAS_STATUS_SUCCESS );

}

template <>
void cublas_axpy( cublas_handle& handle,
  int N, double ALPHA, const double* X, int INCX, double* Y, int INCY 
) {

  auto handle_h = handle.pimpl()->handle;

  auto stat = 
    cublasDaxpy( handle_h, N, &ALPHA, X, INCX, Y, INCY );

  assert( stat == CUBLAS_STATUS_SUCCESS );
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
  cudaError err = cudaGetLastError();
  assert( err == cudaSuccess );
  if ( cudaSuccess != err )
    printf( "AXPBY Error!: %s\n", cudaGetErrorString( err ) );
}

template <>
void axpby_device<double>( 
  int N, double ALPHA, const double* X, int INCX, double BETA, double* Y, int INCY
);

}
