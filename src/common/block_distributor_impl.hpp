//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: David Williams-Young

#pragma once
#include "block_distributor_decl.hpp"

namespace scales {

namespace detail {

template <typename T>
class BlockDistributorImpl {

protected:

  MPI_Comm comm_;
  Int      comm_rank_;
  Int      comm_size_;

  Int M_;
  Int N_;
  Int MBlock_;
  Int NBlock_;
  Int MRem_;
  Int NRem_;
  Int MLocal_;
  Int NLocal_;

public:

  BlockDistributorImpl( MPI_Comm comm, Int M, Int N ) :
    comm_(comm), M_(M), N_(N) {

    MPI_Comm_rank( comm_, &comm_rank_); 
    MPI_Comm_size( comm_, &comm_size_); 

    MBlock_ = M / comm_size_;
    NBlock_ = N / comm_size_;
    MRem_   = M % comm_size_;
    NRem_   = N % comm_size_;

    MLocal_ = MBlock_ + !!(comm_rank_ < ( MRem_ ));
    NLocal_ = NBlock_ + !!(comm_rank_ < ( NRem_ ));

  };
  

  inline auto comm()      const noexcept { return comm_;      }
  inline auto comm_rank() const noexcept { return comm_rank_; }
  inline auto comm_size() const noexcept { return comm_size_; }
  
  inline auto M()      const noexcept { return M_;      }
  inline auto N()      const noexcept { return N_;      }
  inline auto MLocal() const noexcept { return MLocal_; }
  inline auto NLocal() const noexcept { return NLocal_; }

  virtual void redistribute_row_to_col( const NumMat<T>& row_data, NumMat<T>& col_data ) = 0;
  virtual void redistribute_col_to_row( const NumMat<T>& col_data, NumMat<T>& row_data ) = 0;

};

}


template <typename T>
BlockDistributor<T>::BlockDistributor( 
  std::unique_ptr<detail::BlockDistributorImpl<T>>&& impl 
) :
  impl_(std::move(impl)) { }

template <typename T>
BlockDistributor<T>::BlockDistributor() noexcept : impl_(nullptr) { };

//template <typename T>
//BlockDistributor<T>::BlockDistributor( MPI_Comm comm, Int M, Int N ) :
//  BlockDistributor( detail::make_default_host_<T>( comm, M, N ) ) { }
  
template <typename T>
BlockDistributor<T>::~BlockDistributor() noexcept = default;


template <typename T>
BlockDistributor<T>& BlockDistributor<T>::operator=( BlockDistributor&& other ) noexcept {
  impl_ = std::move(other.impl_);
  return (*this);
}

template <typename T>
BlockDistributor<T>::BlockDistributor( BlockDistributor&& other ) noexcept :
  BlockDistributor( std::move(other.impl_) ) { }


template <typename T>
void BlockDistributor<T>::redistribute_row_to_col( const NumMat<T>& row_data, 
                                                         NumMat<T>& col_data ) {
  if( impl_ )
    impl_->redistribute_row_to_col( row_data, col_data );
  else throw std::runtime_error("BlockDistributor Has Not Been Initialized");
}

template <typename T>
void BlockDistributor<T>::redistribute_col_to_row( const NumMat<T>& col_data, 
                                                         NumMat<T>& row_data ) {
  if( impl_ )
    impl_->redistribute_col_to_row( col_data, row_data );
  else throw std::runtime_error("BlockDistributor Has Not Been Initialized");
}







template <typename T>
Int BlockDistributor<T>::M() const {
  if( impl_ ) return impl_->M();
  else throw std::runtime_error("BlockDistributor Has Not Been Initialized");
}
template <typename T>
Int BlockDistributor<T>::N() const {
  if( impl_ ) return impl_->N();
  else throw std::runtime_error("BlockDistributor Has Not Been Initialized");
}
template <typename T>
Int BlockDistributor<T>::MLocal() const {
  if( impl_ ) return impl_->MLocal();
  else throw std::runtime_error("BlockDistributor Has Not Been Initialized");
}
template <typename T>
Int BlockDistributor<T>::NLocal() const {
  if( impl_ ) return impl_->NLocal();
  else throw std::runtime_error("BlockDistributor Has Not Been Initialized");
}






} // namespace scales
