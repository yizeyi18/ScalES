//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Weile Jia

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
