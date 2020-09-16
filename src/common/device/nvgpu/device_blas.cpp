/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Weile Jia

This file is part of DGDFT. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
(2) Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to Lawrence Berkeley National
Laboratory, without imposing a separate written license agreement for such
Enhancements, then you hereby grant the following license: a non-exclusive,
royalty-free perpetual license to install, use, modify, prepare derivative
works, incorporate into other computer software, distribute, and sublicense
such enhancements or derivative works thereof, in binary and source code form.
 */
/// @file device_blas.hpp
/// @brief Thin interface to CUBLAS
/// @date 2020-08-18
#ifdef DEVICE // only used for the GPU version of the PWDFT code. 
#include  "environment.hpp"
#include "device_blas.hpp"

cublasHandle_t hcublas;

inline void __cublas_error(cublasStatus_t status, const char *file, int line, const char *msg)
{
    if(status != CUBLAS_STATUS_SUCCESS)
    {
      float* foo = NULL;
      float bar = foo[0];
      printf("Tried to segfault! %f\n", bar);

        printf("\nCUBLAS Error in %s, line %d: %s\n %s\n", file, line, deviceBLASGetErrorString(status), msg);
        cudaDeviceReset();
        exit(-1);
    }
}

#define CUBLAS_ERROR(status, msg) __cublas_error( status, __FILE__, __LINE__, msg )

namespace dgdft {

/// @namespace DEVICE_BLAS
///
/// @brief Thin interface to CUBLAS
namespace DEVICE_BLAS {

typedef  int               Int;
typedef  cuComplex         scomplex;

void Init(void)
{
    CUBLAS_ERROR( cublasCreate(&hcublas), "Failed to initialze CUBLAS!" );
}

void Destroy(void)
{
    CUBLAS_ERROR( cublasDestroy(hcublas), "Failed to initialze CUBLAS!" );
}

void Gemv ( char transA, Int m, Int n, const double *alpha,
          const double *A, int lda, const double *x, int incx,
          const double *beta, double *y, int incy)
{
    cublasOperation_t trans;
    if (transA == 'n' || transA == 'N') trans = CUBLAS_OP_N;
    if (transA == 't' || transA == 'T') trans = CUBLAS_OP_T;
    if (transA == 'c' || transA == 'C') trans = CUBLAS_OP_C;
    CUBLAS_ERROR( cublasDgemv_v2(hcublas, trans, m, n, alpha, A, lda, x, incx, beta, y, incy ), "cublasDgemv_v2 failed !");
    return;
}


void Gemv ( char transA, Int m, Int n, const float *alpha,
          const float *A, int lda, const float *x, int incx,
          const float *beta, float *y, int incy)
{
    cublasOperation_t trans;
    if (transA == 'n' || transA == 'N') trans = CUBLAS_OP_N;
    if (transA == 't' || transA == 'T') trans = CUBLAS_OP_T;
    if (transA == 'c' || transA == 'C') trans = CUBLAS_OP_C;
    CUBLAS_ERROR( cublasSgemv_v2(hcublas, trans, m, n, alpha, A, lda, x, incx, beta, y, incy ), "cublasSgemv_v2 failed !");
    return;
}


void Gemv ( char transA, Int m, Int n, const cuComplex *alpha,
          const cuComplex *A, int lda, const cuComplex *x, int incx,
          const cuComplex *beta, cuComplex *y, int incy)
{
    cublasOperation_t trans;
    if (transA == 'n' || transA == 'N') trans = CUBLAS_OP_N;
    if (transA == 't' || transA == 'T') trans = CUBLAS_OP_T;
    if (transA == 'c' || transA == 'C') trans = CUBLAS_OP_C;
    CUBLAS_ERROR( cublasCgemv_v2(hcublas, trans, m, n, alpha, A, lda, x, incx, beta, y, incy ), "cublasCgemv_v2 failed !");
    return;
}

void Gemv ( char transA, Int m, Int n, const cuDoubleComplex *alpha,
          const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx,
          const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)
{
    cublasOperation_t trans;
    if (transA == 'n' || transA == 'N') trans = CUBLAS_OP_N;
    if (transA == 't' || transA == 'T') trans = CUBLAS_OP_T;
    if (transA == 'c' || transA == 'C') trans = CUBLAS_OP_C;
    CUBLAS_ERROR( cublasZgemv_v2(hcublas, trans, m, n, alpha, A, lda, x, incx, beta, y, incy ), "cublasZgemv_v2 failed !");
    return;
}

// *********************************************************************
// Level 3 BLAS GEMM 
// *********************************************************************
 void Gemm 
            ( char transA, char transB, Int m, Int n, Int k,
            const float *alpha, const float* A, Int lda, const float* B, Int ldb,
            const float *beta,        float* C, Int ldc )
{
    cublasOperation_t trans1, trans2;
    if (transA == 'n' || transA == 'N') trans1 = CUBLAS_OP_N;
    if (transA == 't' || transA == 'T') trans1 = CUBLAS_OP_T;
    if (transA == 'c' || transA == 'C') trans1 = CUBLAS_OP_C;
    if (transB == 'n' || transB == 'N') trans2 = CUBLAS_OP_N;
    if (transB == 't' || transB == 'T') trans2 = CUBLAS_OP_T;
    if (transB == 'c' || transB == 'C') trans2 = CUBLAS_OP_C;
    CUBLAS_ERROR(cublasSgemm_v2(hcublas, trans1, trans2, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc), " cublasSgemm failed ! ");
    return;
}

 void Gemm 
           ( char transA, char transB, Int m, Int n, Int k,
            const double *alpha, const double* A, Int lda, const double* B, Int ldb,
            double *beta,        double* C, Int ldc )
{
    cublasOperation_t trans1, trans2;
    if (transA == 'n' || transA == 'N') trans1 = CUBLAS_OP_N;
    if (transA == 't' || transA == 'T') trans1 = CUBLAS_OP_T;
    if (transA == 'c' || transA == 'C') trans1 = CUBLAS_OP_C;
    if (transB == 'n' || transB == 'N') trans2 = CUBLAS_OP_N;
    if (transB == 't' || transB == 'T') trans2 = CUBLAS_OP_T;
    if (transB == 'c' || transB == 'C') trans2 = CUBLAS_OP_C;
    CUBLAS_ERROR(cublasDgemm_v2(hcublas, trans1, trans2, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc), " cublasDgemm failed !");
    return;
}
 void Gemm 
          ( char transA, char transB, Int m, Int n, Int k,
            const scomplex *alpha, const scomplex* A, Int lda, const scomplex* B, Int ldb,
            const scomplex *beta,        scomplex* C, Int ldc )
{
    cublasOperation_t trans1, trans2;
    if (transA == 'n' || transA == 'N') trans1 = CUBLAS_OP_N;
    if (transA == 't' || transA == 'T') trans1 = CUBLAS_OP_T;
    if (transA == 'c' || transA == 'C') trans1 = CUBLAS_OP_C;
    if (transB == 'n' || transB == 'N') trans2 = CUBLAS_OP_N;
    if (transB == 't' || transB == 'T') trans2 = CUBLAS_OP_T;
    if (transB == 'c' || transB == 'C') trans2 = CUBLAS_OP_C;
    CUBLAS_ERROR(cublasCgemm_v2(hcublas, trans1, trans2, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc), " cublasCgemm failed !");
    return;
}
 void Gemm 
          ( char transA, char transB, Int m, Int n, Int k,
            const cuDoubleComplex *alpha, const cuDoubleComplex* A, Int lda, const cuDoubleComplex* B, Int ldb,
            const cuDoubleComplex *beta,        cuDoubleComplex* C, Int ldc )
{
    cublasOperation_t trans1, trans2;
    if (transA == 'n' || transA == 'N') trans1 = CUBLAS_OP_N;
    if (transA == 't' || transA == 'T') trans1 = CUBLAS_OP_T;
    if (transA == 'c' || transA == 'C') trans1 = CUBLAS_OP_C;
    if (transB == 'n' || transB == 'N') trans2 = CUBLAS_OP_N;
    if (transB == 't' || transB == 'T') trans2 = CUBLAS_OP_T;
    if (transB == 'c' || transB == 'C') trans2 = CUBLAS_OP_C;
    CUBLAS_ERROR(cublasZgemm_v2(hcublas, trans1, trans2, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc), " cublasZgemm failed !");
    return;
}
 void Gemm
           ( char transA, char transB, Int m, Int n, Int k,
            const double *alpha, double* A[], Int lda, double* B[], Int ldb,
            const double *beta,   double* C[], Int ldc ,Int batchCount)
{
    cublasOperation_t trans1, trans2;
    if (transA == 'n' || transA == 'N') trans1 = CUBLAS_OP_N;
    if (transA == 't' || transA == 'T') trans1 = CUBLAS_OP_T;
    if (transA == 'c' || transA == 'C') trans1 = CUBLAS_OP_C;
    if (transB == 'n' || transB == 'N') trans2 = CUBLAS_OP_N;
    if (transB == 't' || transB == 'T') trans2 = CUBLAS_OP_T;
    if (transB == 'c' || transB == 'C') trans2 = CUBLAS_OP_C;
    
    CUBLAS_ERROR(cublasDgemmBatched(hcublas, trans1, trans2, m, n, k, alpha, const_cast<double**>(A), lda, const_cast<double**>(B), ldb, beta, C, ldc, batchCount), " cublasDgemmBatched failed! ");
    return;
}

void Axpy( int n, const float * alpha, const float * x, int incx, float * y, int incy)
{
    CUBLAS_ERROR( cublasSaxpy(hcublas, n, alpha, x, incx, y, incy), "cublas sAxpy failed! ");
}

void Axpy( int n, const double * alpha, const double * x, int incx, double * y, int incy)
{
    CUBLAS_ERROR( cublasDaxpy(hcublas, n, alpha, x, incx, y, incy), "cublas sAxpy failed! ");
}

void Axpy( int n, const cuComplex * alpha, const cuComplex * x, int incx, cuComplex * y, int incy)
{
    CUBLAS_ERROR( cublasCaxpy(hcublas, n, alpha, x, incx, y, incy), "cublas sAxpy failed! ");
}

void Axpy( int n, const cuDoubleComplex * alpha, const cuDoubleComplex * x, int incx, cuDoubleComplex * y, int incy)
{
    CUBLAS_ERROR( cublasZaxpy(hcublas, n, alpha, x, incx, y, incy), "cublas sAxpy failed! ");
}

 void Scal (int n, const float *alpha, float *x, int incx)
{
    CUBLAS_ERROR( cublasSscal(hcublas, n, alpha, x, incx), "cublas SScal failed! ");
    return;
}
 void Scal (int n, const double *alpha, double *x, int incx)
{
    CUBLAS_ERROR( cublasDscal(hcublas, n, alpha, x, incx), "cublas Dscal failed! ");
    return;
}
 void Scal (int n, const scomplex *alpha, scomplex *x, int incx)
{
    CUBLAS_ERROR( cublasCscal(hcublas, n, alpha, x, incx), "cublas CScal failed! ");
    return;
}
 void Scal (int n, const float *alpha, scomplex *x, int incx)
{
    CUBLAS_ERROR( cublasCsscal(hcublas, n, alpha, x, incx), "cublas CScal failed! ");
    return;
}
 void Scal (int n, const cuDoubleComplex *alpha, cuDoubleComplex *x, int incx)
{
    CUBLAS_ERROR( cublasZscal(hcublas, n, alpha, x, incx), "cublas CScal failed! ");
    return;
}
 void Scal (int n, const double *alpha, cuDoubleComplex *x, int incx)
{
    CUBLAS_ERROR( cublasZdscal(hcublas, n, alpha, x, incx), "cublas CScal failed! ");
    return;
}
 

 void Trsm ( char side, char uplo, char trans, 
             char diag, int m, int n, const float *alpha,  const float *A,  
             int lda, float  *B, int ldb )
{
    cublasOperation_t trans1;
    if (trans == 'n' || trans == 'N') trans1 = CUBLAS_OP_N;
    if (trans == 't' || trans == 'T') trans1 = CUBLAS_OP_T;
    if (trans == 'c' || trans == 'C') trans1 = CUBLAS_OP_C;

    cublasSideMode_t side1;
    if ( side == 'l' || side == 'L' ) side1 = CUBLAS_SIDE_LEFT;
    if ( side == 'r' || side == 'R' ) side1 = CUBLAS_SIDE_RIGHT;

    cublasFillMode_t  uplo1;
    if( uplo == 'u' || uplo == 'U' ) uplo1 = CUBLAS_FILL_MODE_UPPER;
    if( uplo == 'l' || uplo == 'L' ) uplo1 = CUBLAS_FILL_MODE_LOWER;

    cublasDiagType_t diag1;
    if ( diag == 'u' || diag == 'U' ) diag1 = CUBLAS_DIAG_UNIT;
    if ( diag == 'n' || diag == 'N' ) diag1 = CUBLAS_DIAG_NON_UNIT;

    CUBLAS_ERROR( cublasStrsm(hcublas, side1, uplo1, trans1, diag1, m, n, alpha, A, lda, B, ldb ), 
                  " cublas Strsm failed! "); 
    return;
}
 void Trsm ( char side, char uplo, char trans, 
             char diag, int m, int n, const double *alpha, const double *A, 
             int lda, double *B, int ldb )
{   
    cublasOperation_t trans1;
    if (trans == 'n' || trans == 'N') trans1 = CUBLAS_OP_N;
    if (trans == 't' || trans == 'T') trans1 = CUBLAS_OP_T;
    if (trans == 'c' || trans == 'C') trans1 = CUBLAS_OP_C;

    cublasSideMode_t side1;
    if ( side == 'l' || side == 'L' ) side1 = CUBLAS_SIDE_LEFT;
    if ( side == 'r' || side == 'R' ) side1 = CUBLAS_SIDE_RIGHT;

    cublasFillMode_t  uplo1;
    if( uplo == 'u' || uplo == 'U' ) uplo1 = CUBLAS_FILL_MODE_UPPER;
    if( uplo == 'l' || uplo == 'L' ) uplo1 = CUBLAS_FILL_MODE_LOWER;

    cublasDiagType_t diag1;
    if ( diag == 'u' || diag == 'U' ) diag1 = CUBLAS_DIAG_UNIT;
    if ( diag == 'n' || diag == 'N' ) diag1 = CUBLAS_DIAG_NON_UNIT;

    CUBLAS_ERROR( cublasDtrsm(hcublas, side1, uplo1, trans1, diag1, m, n, alpha, A, lda, B, ldb ), 
                  " cublas Dtrsm failed! ");
    return;
}
 void Trsm ( char side, char uplo, char trans, 
             char diag, int m, int n, const cuComplex *alpha, 
             const cuComplex *A, int lda, cuComplex *B, int ldb )
{
    cublasOperation_t trans1;
    if (trans == 'n' || trans == 'N') trans1 = CUBLAS_OP_N;
    if (trans == 't' || trans == 'T') trans1 = CUBLAS_OP_T;
    if (trans == 'c' || trans == 'C') trans1 = CUBLAS_OP_C;

    cublasSideMode_t side1;
    if ( side == 'l' || side == 'L' ) side1 = CUBLAS_SIDE_LEFT;
    if ( side == 'r' || side == 'R' ) side1 = CUBLAS_SIDE_RIGHT;

    cublasFillMode_t  uplo1;
    if( uplo == 'u' || uplo == 'U' ) uplo1 = CUBLAS_FILL_MODE_UPPER;
    if( uplo == 'l' || uplo == 'L' ) uplo1 = CUBLAS_FILL_MODE_LOWER;

    cublasDiagType_t diag1;
    if ( diag == 'u' || diag == 'U' ) diag1 = CUBLAS_DIAG_UNIT;
    if ( diag == 'n' || diag == 'N' ) diag1 = CUBLAS_DIAG_NON_UNIT;

    CUBLAS_ERROR( cublasCtrsm(hcublas, side1, uplo1, trans1, diag1, m, n, alpha, A, lda, B, ldb ), 
                  " cublas Ctrsm failed! ");
    return;
}
 void Trsm ( char side, char uplo, char trans, 
             char diag, int m, int n, const cuDoubleComplex *alpha, 
             const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb )
{
    cublasOperation_t trans1;
    if (trans == 'n' || trans == 'N') trans1 = CUBLAS_OP_N;
    if (trans == 't' || trans == 'T') trans1 = CUBLAS_OP_T;
    if (trans == 'c' || trans == 'C') trans1 = CUBLAS_OP_C;

    cublasSideMode_t side1;
    if ( side == 'l' || side == 'L' ) side1 = CUBLAS_SIDE_LEFT;
    if ( side == 'r' || side == 'R' ) side1 = CUBLAS_SIDE_RIGHT;

    cublasFillMode_t  uplo1;
    if( uplo == 'u' || uplo == 'U' ) uplo1 = CUBLAS_FILL_MODE_UPPER;
    if( uplo == 'l' || uplo == 'L' ) uplo1 = CUBLAS_FILL_MODE_LOWER;

    cublasDiagType_t diag1;
    if ( diag == 'u' || diag == 'U' ) diag1 = CUBLAS_DIAG_UNIT;
    if ( diag == 'n' || diag == 'N' ) diag1 = CUBLAS_DIAG_NON_UNIT;

    CUBLAS_ERROR( cublasZtrsm(hcublas, side1, uplo1, trans1, diag1, m, n, alpha, A, lda, B, ldb ), 
                  " cublas Ztrsm failed! ");
    return;
}

void batched_Gemm( char transA, char transB, int m, int n, int k, const double *alpha, double *A, int lda, double *B, int ldb, const double *beta, double *C, int ldc, int batchCount, int x, int y, int z)
{
	cublasOperation_t trans1, trans2;
	if (transA == 'n' || transA == 'N') trans1 = CUBLAS_OP_N;
	if (transA == 't' || transA == 'T') trans1 = CUBLAS_OP_T;
	if (transA == 'c' || transA == 'C') trans1 = CUBLAS_OP_C;
	if (transB == 'n' || transB == 'N') trans2 = CUBLAS_OP_N;
	if (transB == 't' || transB == 'T') trans2 = CUBLAS_OP_T;
	if (transB == 'c' || transB == 'C') trans2 = CUBLAS_OP_C;
	double ** h_A = (double **) malloc( sizeof(double*) *batchCount);
	double ** h_B = (double **) malloc( sizeof(double*) *batchCount);
	double ** h_C = (double **) malloc( sizeof(double*) *batchCount);
	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i] = A + i* lda;
		h_B[i] = B + i* ldb;
		h_C[i] = C + i*3*x + 3*y + z;
	}
	
	double **d_A, **d_B, **d_C;
	cudaMalloc((void**)&d_A, sizeof(double*) * batchCount);
	cudaMalloc((void**)&d_B, sizeof(double*) * batchCount);
	cudaMalloc((void**)&d_C, sizeof(double*) * batchCount);
	
	cudaMemcpy( d_A, h_A, batchCount *sizeof(double*), cudaMemcpyHostToDevice);
	cudaMemcpy( d_B, h_B, batchCount *sizeof(double*), cudaMemcpyHostToDevice);
	cudaMemcpy( d_C, h_C, batchCount *sizeof(double*), cudaMemcpyHostToDevice);
	
	cublasDgemmBatched(hcublas, trans1, trans2, m, n, k, alpha, (const double**)(d_A), lda, (const double**)(d_B), ldb, beta, d_C, ldc, batchCount);

	free(h_A);
	free(h_B);
	free(h_C);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

void batched_Gemm6( char transA, char transB, int m, int n, int k, const double *alpha, double *A, int lda, double *B, int ldb, const double *beta, double *C, int ldc, int batchCount, int x, int y, int z, 
 double *A2, double *B2, double *C2, int x2, int y2, int z2,
 double *A3, double *B3, double *C3, int x3, int y3, int z3,
 double *A4, double *B4, double *C4, int x4, int y4, int z4,
 double *A5, double *B5, double *C5, int x5, int y5, int z5,
 double *A6, double *B6, double *C6, int x6, int y6, int z6)
{
	cublasOperation_t trans1, trans2;
	if (transA == 'n' || transA == 'N') trans1 = CUBLAS_OP_N;
	if (transA == 't' || transA == 'T') trans1 = CUBLAS_OP_T;
	if (transA == 'c' || transA == 'C') trans1 = CUBLAS_OP_C;
	if (transB == 'n' || transB == 'N') trans2 = CUBLAS_OP_N;
	if (transB == 't' || transB == 'T') trans2 = CUBLAS_OP_T;
	if (transB == 'c' || transB == 'C') trans2 = CUBLAS_OP_C;

	double ** h_A = (double **) malloc( sizeof(double*) *6*batchCount);
	double ** h_B = (double **) malloc( sizeof(double*) *6*batchCount);
	double ** h_C = (double **) malloc( sizeof(double*) *6*batchCount);

	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i] = A + i* lda;
		h_B[i] = B + i* ldb;
		h_C[i] = C + i*3*x + 3*y + z;
	}
	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i+batchCount] = A2 + i* lda;
		h_B[i+batchCount] = B2 + i* ldb;
		h_C[i+batchCount] = C2 + i*3*x2 + 3*y2 + z2;
	}
	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i+2*batchCount] = A3 + i* lda;
		h_B[i+2*batchCount] = B3 + i* ldb;
		h_C[i+2*batchCount] = C3 + i*3*x3 + 3*y3 + z3;
	}
	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i+3*batchCount] = A4 + i* lda;
		h_B[i+3*batchCount] = B4 + i* ldb;
		h_C[i+3*batchCount] = C4 + i*3*x4 + 3*y4 + z4;
	}
	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i+4*batchCount] = A5 + i* lda;
		h_B[i+4*batchCount] = B5 + i* ldb;
		h_C[i+4*batchCount] = C5 + i*3*x5 + 3*y5 + z5;
	}
	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i+5*batchCount] = A6 + i* lda;
		h_B[i+5*batchCount] = B6 + i* ldb;
		h_C[i+5*batchCount] = C6 + i*3*x6 + 3*y6 + z6;
	}
	
	double **d_A, **d_B, **d_C;
	cudaMalloc((void**)&d_A, sizeof(double*) * 6*batchCount);
	cudaMalloc((void**)&d_B, sizeof(double*) * 6*batchCount);
	cudaMalloc((void**)&d_C, sizeof(double*) * 6*batchCount);
	
	cudaMemcpy( d_A, h_A, batchCount *6*sizeof(double*), cudaMemcpyHostToDevice);
	cudaMemcpy( d_B, h_B, batchCount *6*sizeof(double*), cudaMemcpyHostToDevice);
	cudaMemcpy( d_C, h_C, batchCount *6*sizeof(double*), cudaMemcpyHostToDevice);
	
	cublasDgemmBatched(hcublas, trans1, trans2, m, n, k, alpha, (const double**)(d_A), lda, (const double**)(d_B), ldb, beta, d_C, ldc, 6*batchCount);

	free(h_A);
	free(h_B);
	free(h_C);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}


void batched_Gemm6( char transA, char transB, int m, int n, int k, const cuDoubleComplex *alpha, cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc, int batchCount, int x, int y, int z, 
 cuDoubleComplex *A2, cuDoubleComplex *B2, cuDoubleComplex *C2, int x2, int y2, int z2,
 cuDoubleComplex *A3, cuDoubleComplex *B3, cuDoubleComplex *C3, int x3, int y3, int z3,
 cuDoubleComplex *A4, cuDoubleComplex *B4, cuDoubleComplex *C4, int x4, int y4, int z4,
 cuDoubleComplex *A5, cuDoubleComplex *B5, cuDoubleComplex *C5, int x5, int y5, int z5,
 cuDoubleComplex *A6, cuDoubleComplex *B6, cuDoubleComplex *C6, int x6, int y6, int z6)
{
    cublasOperation_t trans1, trans2;
    if (transA == 'n' || transA == 'N') trans1 = CUBLAS_OP_N;
    if (transA == 't' || transA == 'T') trans1 = CUBLAS_OP_T;
    if (transA == 'c' || transA == 'C') trans1 = CUBLAS_OP_C;
    if (transB == 'n' || transB == 'N') trans2 = CUBLAS_OP_N;
    if (transB == 't' || transB == 'T') trans2 = CUBLAS_OP_T;
    if (transB == 'c' || transB == 'C') trans2 = CUBLAS_OP_C;
	cuDoubleComplex ** h_A = (cuDoubleComplex **) malloc( sizeof(cuDoubleComplex*) *6*batchCount);
	cuDoubleComplex ** h_B = (cuDoubleComplex **) malloc( sizeof(cuDoubleComplex*) *6*batchCount);
	cuDoubleComplex ** h_C = (cuDoubleComplex **) malloc( sizeof(cuDoubleComplex*) *6*batchCount);

	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i] = A + i* lda;
		h_B[i] = B + i* ldb;
		h_C[i] = C + i*3*x + 3*y + z;
	}
	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i+batchCount] = A2 + i* lda;
		h_B[i+batchCount] = B2 + i* ldb;
		h_C[i+batchCount] = C2 + i*3*x2 + 3*y2 + z2;
	}
	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i+2*batchCount] = A3 + i* lda;
		h_B[i+2*batchCount] = B3 + i* ldb;
		h_C[i+2*batchCount] = C3 + i*3*x3 + 3*y3 + z3;
	}
	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i+3*batchCount] = A4 + i* lda;
		h_B[i+3*batchCount] = B4 + i* ldb;
		h_C[i+3*batchCount] = C4 + i*3*x4 + 3*y4 + z4;
	}
	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i+4*batchCount] = A5 + i* lda;
		h_B[i+4*batchCount] = B5 + i* ldb;
		h_C[i+4*batchCount] = C5 + i*3*x5 + 3*y5 + z5;
	}
	for(int i = 0; i < batchCount; i++ )
	{
		h_A[i+5*batchCount] = A6 + i* lda;
		h_B[i+5*batchCount] = B6 + i* ldb;
		h_C[i+5*batchCount] = C6 + i*3*x6 + 3*y6 + z6;
	}
	
	cuDoubleComplex **d_A, **d_B, **d_C;
	cudaMalloc((void**)&d_A, sizeof(cuDoubleComplex*) * 6*batchCount);
	cudaMalloc((void**)&d_B, sizeof(cuDoubleComplex*) * 6*batchCount);
	cudaMalloc((void**)&d_C, sizeof(cuDoubleComplex*) * 6*batchCount);
	
	cudaMemcpy( d_A, h_A, batchCount *6*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
	cudaMemcpy( d_B, h_B, batchCount *6*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
	cudaMemcpy( d_C, h_C, batchCount *6*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
	
	cublasZgemmBatched(hcublas, trans1, trans2, m, n, k, alpha, (const cuDoubleComplex**)(d_A), lda, (const cuDoubleComplex**)(d_B), ldb, beta, d_C, ldc, 6*batchCount);

	free(h_A);
	free(h_B);
	free(h_C);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
  
void batched_least_square(Int m, Int n, Int nrhs, cuDoubleComplex *A, Int lda, cuDoubleComplex *C, Int ldc, Int batchSize , int maxDim)
{
	int info;
	int *devInfoArray;
        cudaMalloc( (void**)&devInfoArray, sizeof(int*) * batchSize );

        cuDoubleComplex ** h_A = ( cuDoubleComplex**) malloc ( sizeof(cuDoubleComplex*) * batchSize );
        cuDoubleComplex ** h_C = ( cuDoubleComplex**) malloc ( sizeof(cuDoubleComplex*) * batchSize );

        for( int i = 0; i < batchSize; i++)
        {
                h_A[i] = A + i * lda * maxDim;
                h_C[i] = C + i * ldc * maxDim;
        }

        cuDoubleComplex **d_A, **d_C;
        cudaMalloc( (void**)&d_A, sizeof(cuDoubleComplex*) * batchSize );
        cudaMalloc( (void**)&d_C, sizeof(cuDoubleComplex*) * batchSize );
        cudaMemcpy( d_A, h_A, batchSize*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
        cudaMemcpy( d_C, h_C, batchSize*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);

        CUBLAS_ERROR( cublasZgelsBatched ( hcublas, CUBLAS_OP_N, m, n, nrhs, d_A, lda, d_C, ldc, &info, devInfoArray, batchSize ), "cublasZgelsBatched failed!" );
	
	if( info < 0) 
		std::cout << " Error,"<< -info << "th parameter is passed wrong. cublasZgelsBatched"<< std::endl;

        free(h_A);
        free(h_C);

        cudaFree(d_A);
        cudaFree(d_C);
        cudaFree(devInfoArray);

        return;
}

} // namespace cublas
} // namespace dgdft

#endif
