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
#include  "environment.hpp"

#include "cublas.hpp"
cublasHandle_t hcublas;

inline void __cublas_error(cublasStatus_t status, const char *file, int line, const char *msg)
{
    if(status != CUBLAS_STATUS_SUCCESS)
    {
      float* foo = NULL;
      float bar = foo[0];
      printf("Tried to segfault! %f\n", bar);

        printf("\nCUBLAS Error in %s, line %d: %s\n %s\n", file, line, cublasGetErrorString(status), msg);
        cudaDeviceReset();
        exit(-1);
    }
}

#define CUBLAS_ERROR(status, msg) __cublas_error( status, __FILE__, __LINE__, msg )

namespace dgdft {

/// @namespace cublas
///
/// @brief Thin interface to CUBLAS
namespace cublas {

typedef  int               Int;
typedef  cuComplex         scomplex;
typedef  cuDoubleComplex   dcomplex;

void Init(void)
{
    CUBLAS_ERROR( cublasCreate(&hcublas), "Failed to initialze CUBLAS!" );
}

void Destroy(void)
{
    CUBLAS_ERROR( cublasDestroy(hcublas), "Failed to initialze CUBLAS!" );
}
// *********************************************************************
// Level 3 BLAS GEMM 
// *********************************************************************
 void Gemm 
            ( cublasOperation_t transA, cublasOperation_t transB, Int m, Int n, Int k,
            const float *alpha, const float* A, Int lda, const float* B, Int ldb,
            const float *beta,        float* C, Int ldc )
{
    CUBLAS_ERROR(cublasSgemm_v2(hcublas, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc), " cublasSgemm failed ! ");
    return;
}

 void Gemm 
           ( cublasOperation_t transA, cublasOperation_t transB, Int m, Int n, Int k,
            const double *alpha, const double* A, Int lda, const double* B, Int ldb,
            double *beta,        double* C, Int ldc )
{
    CUBLAS_ERROR(cublasDgemm_v2(hcublas, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc), " cublasDgemm failed !");
    return;
}
 void Gemm 
          ( cublasOperation_t transA, cublasOperation_t transB, Int m, Int n, Int k,
            const scomplex *alpha, const scomplex* A, Int lda, const scomplex* B, Int ldb,
            const scomplex *beta,        scomplex* C, Int ldc )
{
    CUBLAS_ERROR(cublasCgemm_v2(hcublas, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc), " cublasCgemm failed !");
    return;
}
 void Gemm 
          ( cublasOperation_t transA, cublasOperation_t transB, Int m, Int n, Int k,
            const dcomplex *alpha, const dcomplex* A, Int lda, const dcomplex* B, Int ldb,
            const dcomplex *beta,        dcomplex* C, Int ldc )
{
    CUBLAS_ERROR(cublasZgemm_v2(hcublas, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc), " cublasZgemm failed !");
    return;
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
 void Scal (int n, const dcomplex *alpha, dcomplex *x, int incx)
{
    CUBLAS_ERROR( cublasZscal(hcublas, n, alpha, x, incx), "cublas CScal failed! ");
    return;
}
 void Scal (int n, const double *alpha, dcomplex *x, int incx)
{
    CUBLAS_ERROR( cublasZdscal(hcublas, n, alpha, x, incx), "cublas CScal failed! ");
    return;
}

} // namespace cublas
} // namespace dgdft

#endif
