/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Weile Jia and Lin Lin

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
/// @file cublas.hpp
/// @brief Thin interface to CUBLAS
/// @date 2016-10-21
#ifdef GPU  // only used for the GPU version of the PWDFT code. 
#ifndef _CUBLAS_HPP_
#define _CUBLAS_HPP_
#include  "environment.hpp"

#include <cuda_runtime.h>
#include <cuda.h>
#include "cublas_v2.h"
#include "cuda_errors.hpp"
extern cublasHandle_t hcublas;

namespace dgdft {

/// @namespace cublas
///
/// @brief Thin interface to CUBLAS
namespace cublas {

typedef  int               Int;
typedef  cuComplex         scomplex;
typedef  cuDoubleComplex   dcomplex;

void Init(void);

void Destroy(void);
// *********************************************************************
// Level 3 BLAS GEMM 
// *********************************************************************
 void Gemm 
            ( cublasOperation_t transA, cublasOperation_t transB, Int m, Int n, Int k,
            const float *alpha, const float* A, Int lda, const float* B, Int ldb,
            const float *beta,        float* C, Int ldc );

 void Gemm 
           ( cublasOperation_t transA, cublasOperation_t transB, Int m, Int n, Int k,
            const double *alpha, const double* A, Int lda, const double* B, Int ldb,
            double *beta,        double* C, Int ldc );
 void Gemm 
          ( cublasOperation_t transA, cublasOperation_t transB, Int m, Int n, Int k,
            const scomplex *alpha, const scomplex* A, Int lda, const scomplex* B, Int ldb,
            const scomplex *beta,        scomplex* C, Int ldc );
 void Gemm 
          ( cublasOperation_t transA, cublasOperation_t transB, Int m, Int n, Int k,
            const dcomplex *alpha, const dcomplex* A, Int lda, const dcomplex* B, Int ldb,
            const dcomplex *beta,        dcomplex* C, Int ldc );
 void Scal (int n, const float *alpha, float *x, int incx);
 void Scal (int n, const double *alpha, double *x, int incx);
 void Scal (int n, const scomplex *alpha, scomplex *x, int incx);
 void Scal (int n, const float *alpha, scomplex *x, int incx);
 void Scal (int n, const dcomplex *alpha, dcomplex *x, int incx);
 void Scal (int n, const double *alpha, dcomplex *x, int incx);

 void Trsm ( cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, 
             cublasDiagType_t diag, int m, int n, const float *alpha,  const float *A,  
             int lda, float  *B, int ldb );
 void Trsm ( cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, 
             cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, 
             int lda, double *B, int ldb );
 void Trsm ( cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, 
             cublasDiagType_t diag, int m, int n, const cuComplex *alpha, 
             const cuComplex *A, int lda, cuComplex *B, int ldb );
 void Trsm ( cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, 
             cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, 
             const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb );
} // namespace cublas
} // namespace dgdft

#endif
#endif
