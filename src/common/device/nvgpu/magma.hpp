/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Weile Jia

This file is part of ScalES. All rights reserved.

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
/// @file MAGMA.hpp
/// @brief Thin interface to MAGMA
/// @date 2020-08-21


/// @date 2020-8-7 
/// MAGMA interface is not used in most CUDA code anymore, however, it might
/// be useful in the future AMD GPU implementation. Thus, I will keep this interface
/// for future purposes. 

/// Note, only will be called in SYEVD, POTRF. 
#ifdef GPU  // only used for the GPU version of the PWDFT code. 
#ifndef _MAGMA_HPP_
#define _MAGMA_HPP_
#include  "environment.hpp"

#include "magma.h"

namespace scales {

/// @namespace MAGMA
///
/// @brief Thin interface to MAGMA
namespace MAGMA{

typedef  int               Int;
typedef  cuComplex         scomplex;
typedef  cuDoubleComplex   dcomplex;

void Init(void);

void Destroy(void);

void Potrf( char uplo, Int n, const double * A, Int lda );
void Potrf( char uplo, Int n, const cuDoubleComplex* A, Int lda );

void Syevd( char jobz, char uplo, Int n, double *A, Int lda, double *eigs);
void Syevd( char jobz, char uplo, Int n, cuDoubleComplex *A, Int lda, double *eigs);

void Lacpy(char uplo, Int m, Int n, const double * A , Int lda, double *B, Int ldb);

void Zgels( Int m, Int n, Int nrhs, cuDoubleComplex * A, Int lda, cuDoubleComplex * B, Int ldb);

} // namespace MAGMA
} // namespace scales

#endif
#endif
