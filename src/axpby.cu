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

template <typename T>
__global__ void axpby_combined_kernel( const size_t n, const T c, const T e, const T sigma, const T sigma_new, 
	const T* X, const size_t INCX, const T* M, const size_t INCM, T* Y, const size_t INCY )
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid < n ) {
	
		const int tid_inc_y = tid * INCY;
		const int tid_inc_x = tid * INCX;

		if(M == NULL) {
			Y[ tid_inc_y ] = ( Y[ tid_inc_y ] + (c * X[ tid_inc_x]) ) * (sigma/e);
		}
		else {
			const int tid_inc_m = tid * INCM;
			Y[ tid_inc_y ] = ( Y[ tid_inc_y ] + (c * X[ tid_inc_x]) ) * (2.0 * sigma_new/e) - ((sigma * sigma_new) * M[ tid_inc_m]);
		}
	}
}


//Compute R = (M - c * Y) * (2 * sigma_new / e) - (sigma * sigma_new) * X
//if X is NULL
//Compute R = (M - c * Y) * (sigma / e)
//TODO: change parameters to DistVec<Index3, DblNumMat, ElemPrtn>  pluck_Yt; 
template <typename T>
void axpby_combined_device( 
  int N, const T c, const T e, const T sigma, const T sigma_new, 
	const T* X, int INCX, const T* M, int INCM, T* Y, int INCY
) {

  auto div = std::div( N, 1024 );
  
  axpby_combined_kernel<T><<< div.quot + !!div.rem, 1024 >>>( 
			N, c, e, sigma, sigma_new, X, INCX, M, INCM, Y, INCY 
  );
  CUDA_THROW( cudaGetLastError() );
}
template
void axpby_combined_device( 
  int N, double c, double e, double sigma, double sigma_new, 
	const double* X, int INCX, const double* M, int INCM, double* Y, int INCY
); 

}
}
