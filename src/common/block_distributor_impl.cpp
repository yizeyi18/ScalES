//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: David Williams-Young

#include "block_distributor_impl.h"
#include "block_distributor_factory.h"

namespace scales {

#define bdist_impl(T) \
template class BlockDistributor<T>; \
template BlockDistributor<T> make_block_distributor( BlockDistAlg alg,\
                                            MPI_Comm comm, \
                                            Int M, \
                                            Int N );


bdist_impl(double);
// TODO Complex


} // namespace scales
