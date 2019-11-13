/// @file cuda_blas.hpp
/// @brief Thin interface to CUDA BLAS
/// @date 2019-09-9
#ifndef _CUDA_BLAS_HPP_
#define _CUDA_BLAS_HPP_


#include <complex>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
//namespace dgdft {

/// @namespace cudablas
///
//namespace cudablas {

typedef  int                    Int;
typedef  std::complex<float>    scomplex;
typedef  std::complex<double>   dcomplex;


// *********************************************************************
// Level 3 BLAS                                                  
// *********************************************************************
	cublasOperation_t cublasOpFromChar(char op);
template <class T>
__global__ void axpby_kernel( const size_t n, const T alpha, const T* X, const size_t INCX, const T beta, T* Y, const size_t INCY ) {

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid < n ) {
		const int tid_inc_y = tid * INCY;
		const int tid_inc_x = tid * INCX;
		Y[ tid_inc_y ] = beta * Y[ tid_inc_y ] + alpha * X[ tid_inc_x ];
	}
}




template <class T>
void axpby_device( const size_t n, const T alpha, const T* X, const size_t INCX, const T beta, T* Y, const size_t INCY, cudaStream_t stream = 0) {
//	cudaDeviceSynchronize();
	axpby_kernel<T><<< std::ceil( n / 1024.0 ), 1024, 0, stream >>>( n, alpha, X, INCX, beta, Y, INCY );
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
    printf( "AXPBY Error!: %s\n", cudaGetErrorString( err ) );

}
//} // namespace cudablas
//} // namespace dgdft
#endif // _CUDA_BLAS_HPP_
