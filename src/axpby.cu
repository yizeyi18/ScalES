#include <axpby.hpp>
#include <exceptions.hpp>

namespace dgdft {
namespace device {

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
}
