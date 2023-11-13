//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Weile Jia

/// @file device_utility.hpp
/// @brief DEVICE_UTILITY subroutines.
/// @date 2020-08-12
#ifdef DEVICE
#ifndef _DEVICE_UTILITY_HPP_ 
#define _DEVICE_UTILITY_HPP_

#include  <stdlib.h>
#include  "domain.h"
#include  "environment.h"
#include  "numvec_impl.hpp"
#include  "nummat_impl.hpp"
#include  "numtns_impl.hpp"
#include  "sparse_matrix_impl.hpp"
#include  "device_nummat_impl.hpp"
#include  "device_numvec_impl.hpp"
namespace scales{

void device_AlltoallBackward( deviceDblNumMat& A, deviceDblNumMat& B, MPI_Comm comm );
void device_AlltoallForward ( deviceDblNumMat& A, deviceDblNumMat& B, MPI_Comm comm );

} // namespace scales
#endif // _DEVICE_UTILITY_HPP_
#endif
