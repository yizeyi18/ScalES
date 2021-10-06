//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin

/// @file scales.hpp
/// @brief Main header file for ScalES.
/// @date 2012-08-01
#ifndef _ScalES_HPP_
#define _ScalES_HPP_

#include  "environment.hpp"
#include    "blas.hpp"
#ifdef GPU
#include    "cublas.hpp"
#endif
#include    "lapack.hpp"
#include    "scalapack.hpp"
#include  "mpi_interf.hpp"
#include  "numvec_impl.hpp"
#include  "nummat_impl.hpp"
#include  "numtns_impl.hpp"
#include  "tinyvec_impl.hpp"
#include  "distvec_impl.hpp"
#include  "sparse_matrix_impl.hpp"
#include  "utility.hpp"

#include  "domain.hpp"
#include  "fourier.hpp"
#include  "hamiltonian.hpp"
#include  "spinor.hpp"
#include  "periodtable.hpp"
#include  "esdf.hpp"
#include  "eigensolver.hpp"
#include  "utility.hpp"
//#include  "hamiltonian_dg.hpp"
#include  "scf.hpp"
#include  "scf_dg.hpp"
#include  "iondynamics.hpp"
//#include  "tddft.hpp"


#endif // _ScalES_HPP_


