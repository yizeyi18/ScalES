#include "cuda_blas.hpp"
#include <iostream>


//----------------------------------------------------------------------------//
// Level 3 BLAS                                                               //
//----------------------------------------------------------------------------//
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
	}
}

template <class T>
__global__ void axpby_kernel( const size_t n, const T alpha, const T* X, const size_t INCX, const T beta, T* Y, const size_t INCY ) {
	//__global__ void axpby_kernel( const size_t n, const double alpha, const double* X, const size_t INCX, const double beta, double* Y, const size_t INCY ) {

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid < n ) {
		const int tid_inc_y = tid * INCY;
		const int tid_inc_x = tid * INCX;
		Y[ tid_inc_y ] = beta * Y[ tid_inc_y ] + beta * X[ tid_inc_x ];
	}
}




template <class T>
void axpby_device( const size_t n, const T alpha, const T* X, const size_t INCX, const T beta, T* Y, const size_t INCY, cudaStream_t stream ) {
	//void axpby_device( const size_t n, const double alpha, const double* X, const size_t INCX, const double beta, double* Y, const size_t INCY, cudaStream_t stream ) {

	axpby_kernel<<< std::ceil( n / 1024 ), 1024, 0, stream >>>( n, alpha, X, INCX, beta, Y, INCY );

}
