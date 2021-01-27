//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Weile Jia

/// @file device_solver.hpp
/// @brief Thin interface to device_solver
/// @date 2020-08-21

#ifdef DEVICE  
#ifndef _DEVICE_SOLVER_HPP_
#define _DEVICE_SOLVER_HPP_
#include  "environment.hpp"
#include <cuda_runtime.h>
#include <cusolverDn.h>

extern cusolverDnHandle_t cusolverH;

namespace scales {

/// @namespace device_solver
///
/// @brief Thin interface to device_solver
namespace device_solver{

typedef  int               Int;
typedef  cuComplex         scomplex;
typedef  cuDoubleComplex   dcomplex;

void Init(void);

void Destroy(void);

void Potrf( char uplo, Int n, double * A, Int lda );

void Potrf( char uplo, Int n, cuDoubleComplex * A, Int lda );

void Syevd( char jobz, char uplo, Int n, double *A, Int lda, double *eigs);

void Syevd( char jobz, char uplo, Int n, cuDoubleComplex *A, Int lda, double *eigs);

void Lacpy(char uplo, Int m, Int n, const double * A , Int lda, double *B, Int ldb);


} // namespace device_solver
} // namespace scales

#endif
#endif
