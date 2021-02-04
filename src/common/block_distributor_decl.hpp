//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: David Williams-Young

#pragma once

#include "environment.hpp"
#include "nummat_decl.hpp"
#include "numvec_decl.hpp"
#include <memory>

namespace scales {


// Block Distribution Algorithms
enum class BlockDistAlg {
  HostGeneric,
  HostOptPack
};




namespace detail {

template <typename T>
class BlockDistributorImpl;

}

template <typename T>
class BlockDistributor {

  std::unique_ptr< detail::BlockDistributorImpl<T> > impl_;

public:

  BlockDistributor() noexcept;
  BlockDistributor( std::unique_ptr< detail::BlockDistributorImpl<T> >&& impl );
  //BlockDistributor( MPI_Comm comm, Int M, Int N );
  ~BlockDistributor() noexcept;

  BlockDistributor( const BlockDistributor& ) = delete;
  BlockDistributor( BlockDistributor&& ) noexcept;

  BlockDistributor& operator=( const BlockDistributor& ) = delete;
  BlockDistributor& operator=( BlockDistributor&& ) noexcept;

  void redistribute_row_to_col( const NumMat<T>& row_data, NumMat<T>& col_data );
  void redistribute_col_to_row( const NumMat<T>& col_data, NumMat<T>& row_data );

  Int M() const;
  Int N() const;
  Int MLocal() const;
  Int NLocal() const;

};


template <typename T>
BlockDistributor<T> make_block_distributor( BlockDistAlg alg, MPI_Comm comm, Int M, Int N );

} // namespace scales
