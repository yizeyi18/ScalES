//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Weile Jia

/// @file DEVICE_BLAS.hpp
/// @brief Thin interface to CUBLAS
/// @date 2020-08-18
#ifdef DEVICE // only used for the GPU version of the PWDFT code. 
#ifndef _DEVICE_BLAS_HPP_
#define _DEVICE_BLAS_HPP_

#include  "environment.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "device_error.hpp"

extern cublasHandle_t hcublas;

namespace scales {

/// @namespace cublas
///
/// @brief Thin interface to CUBLAS
namespace DEVICE_BLAS {

typedef  int               Int;
typedef  cuComplex         scomplex;
typedef  cuDoubleComplex   dcomplex;

void Init(void);

void Destroy(void);
void Gemv ( char transA, Int m, Int n, const double *alpha,
          const double *A, int lda, const double *x, int incx,
          const double *beta, double *y, int incy);
void Gemv ( char transA, Int m, Int n, const float *alpha,
          const float *A, int lda, const float *x, int incx,
          const float *beta, float *y, int incy);
void Gemv ( char transA, Int m, Int n, const cuComplex *alpha,
          const cuComplex *A, int lda, const cuComplex *x, int incx,
          const cuComplex *beta, cuComplex *y, int incy);
void Gemv ( char transA, Int m, Int n, const cuDoubleComplex *alpha,
          const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx,
          const cuDoubleComplex *beta, cuDoubleComplex *y, int incy);
// *********************************************************************
// Level 3 BLAS GEMM 
// *********************************************************************
 void Gemm 
            ( char transA, char transB, Int m, Int n, Int k,
            const float *alpha, const float* A, Int lda, const float* B, Int ldb,
            const float *beta,        float* C, Int ldc );

 void Gemm 
           ( char transA, char transB, Int m, Int n, Int k,
            const double *alpha, const double* A, Int lda, const double* B, Int ldb,
            double *beta,              double* C, Int ldc );
 void Gemm 
          ( char transA, char transB, Int m, Int n, Int k,
            const scomplex *alpha, const scomplex* A, Int lda, const scomplex* B, Int ldb,
            const scomplex *beta,        scomplex* C, Int ldc );
 void Gemm 
          ( char transA, char transB, Int m, Int n, Int k,
            const dcomplex *alpha, const dcomplex* A, Int lda, const dcomplex* B, Int ldb,
            const dcomplex *beta,        dcomplex* C, Int ldc );
 void Gemm
           ( char transA, char transB, Int m, Int n, Int k,
            const double *alpha, double* A[], Int lda, double* B[], Int ldb,
            const double *beta,          double* C[], Int ldc, Int batchCount);

void batched_Gemm( char transA, char transB, int m, int n, int k, const double *alpha, double *A, int lda, double *B, int ldb, const double *beta, double *C, int ldc, int batchCount, int x, int y,int z);
void batched_Gemm6( char transA, char transB, int m, int n, int k, const double *alpha, double *A, int lda, double *B, int ldb, const double *beta, double *C, int ldc, int batchCount, int x, int y, int z, 
 double *A2, double *B2, double *C2, int x2, int y2, int z2,
 double *A3, double *B3, double *C3, int x3, int y3, int z3,
 double *A4, double *B4, double *C4, int x4, int y4, int z4,
 double *A5, double *B5, double *C5, int x5, int y5, int z5,
 double *A6, double *B6, double *C6, int x6, int y6, int z6);
void batched_Gemm6( char transA, char transB, int m, int n, int k, const cuDoubleComplex *alpha, cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc, int batchCount, int x, int y, int z, 
 cuDoubleComplex *A2, cuDoubleComplex *B2, cuDoubleComplex *C2, int x2, int y2, int z2,
 cuDoubleComplex *A3, cuDoubleComplex *B3, cuDoubleComplex *C3, int x3, int y3, int z3,
 cuDoubleComplex *A4, cuDoubleComplex *B4, cuDoubleComplex *C4, int x4, int y4, int z4,
 cuDoubleComplex *A5, cuDoubleComplex *B5, cuDoubleComplex *C5, int x5, int y5, int z5,
 cuDoubleComplex *A6, cuDoubleComplex *B6, cuDoubleComplex *C6, int x6, int y6, int z6);


 void Scal (int n, const float *alpha, float *x, int incx);
 void Scal (int n, const double *alpha, double *x, int incx);
 void Scal (int n, const scomplex *alpha, scomplex *x, int incx);
 void Scal (int n, const float *alpha, scomplex *x, int incx);
 void Scal (int n, const dcomplex *alpha, dcomplex *x, int incx);
 void Scal (int n, const double *alpha, dcomplex *x, int incx);

void Axpy( int n, const cuDoubleComplex * alpha, const cuDoubleComplex * x, int incx, cuDoubleComplex * y, int incy);
void Axpy( int n, const cuComplex * alpha, const cuComplex * x, int incx, cuComplex * y, int incy);
void Axpy( int n, const double * alpha, const double * x, int incx, double * y, int incy);
void Axpy( int n, const float * alpha, const float * x, int incx, float * y, int incy);

 void Trsm ( char side, char uplo, char trans, 
             char diag, int m, int n, const float *alpha,  const float *A,  
             int lda, float  *B, int ldb );
 void Trsm ( char side, char uplo, char trans, 
             char diag, int m, int n, const double *alpha, const double *A, 
             int lda, double *B, int ldb );
 void Trsm ( char side, char uplo, char trans, 
             char diag, int m, int n, const cuComplex *alpha, 
             const cuComplex *A, int lda, cuComplex *B, int ldb );
 void Trsm ( char side, char uplo, char trans, 
             char diag, int m, int n, const cuDoubleComplex *alpha, 
             const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb );
void batched_least_square(Int m, Int n, Int nrhs, cuDoubleComplex *A, Int lda, cuDoubleComplex *C, Int ldc, Int batchSize , int maxDim);
} // namespace device_blas
} // namespace scales

#endif
#endif
