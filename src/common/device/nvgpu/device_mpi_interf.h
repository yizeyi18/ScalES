//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin and Weile Jia

/// @file device_mpi_interf.h
/// @brief Interface with MPI to facilitate communication.
/// @date 2020-08-23
#ifndef _DEVICE_MPI_INTERF_HPP_
#define _DEVICE_MPI_INTERF_HPP_

#include "environment.h"
#include "cuda.h"
#include "cuda_runtime.h"
namespace scales{

/// @namespace mpi
///
/// @brief Interface with MPI to facilitate communication.
namespace device_mpi{

  void setDevice(MPI_Comm comm);

} // namespace mpi

} // namespace scales



#endif // _DEVICE_MPI_INTERF_HPP_

