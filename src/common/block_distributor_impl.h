//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: David Williams-Young

#pragma once
#include "block_distributor_decl.h"

namespace scales {

namespace detail {

template <typename T>
class BlockDistributorImpl {

protected:

  MPI_Comm comm_;
  Int      comm_rank_;
  Int      comm_size_;

  Int M_;                                                      //M/N是做什么的？待补
  Int N_;
  Int MBlock_;
  Int NBlock_;
  Int MRem_;                                                   //Rem==Remain，block填不上的部分
  Int NRem_;
  Int MLocal_;
  Int NLocal_;

public:

  BlockDistributorImpl( MPI_Comm comm, Int M, Int N ) :        //构造函数
    comm_(comm), M_(M), N_(N) {

    MPI_Comm_rank( comm_, &comm_rank_); 
    MPI_Comm_size( comm_, &comm_size_); 

    MBlock_ = M / comm_size_;
    NBlock_ = N / comm_size_;
    MRem_   = M % comm_size_;
    NRem_   = N % comm_size_;

    MLocal_ = MBlock_ + !!(comm_rank_ < ( MRem_ ));            //!!不是单独的算符，它只是非非
    NLocal_ = NBlock_ + !!(comm_rank_ < ( NRem_ ));            //把真值搞成1用的，叠了一个bool->int的隐式转换，感觉直接(int)强转也行？

  };//BlockDistributorImpl()
  

  inline auto comm()      const noexcept { return comm_;      }
  inline auto comm_rank() const noexcept { return comm_rank_; }
  inline auto comm_size() const noexcept { return comm_size_; }
  
  inline auto M()      const noexcept { return M_;      }
  inline auto N()      const noexcept { return N_;      }
  inline auto MLocal() const noexcept { return MLocal_; }
  inline auto NLocal() const noexcept { return NLocal_; }

  virtual void redistribute_row_to_col( const NumMat<T>& row_data, NumMat<T>& col_data ) = 0; //实现见host_optpack_block_distributor.h
  virtual void redistribute_col_to_row( const NumMat<T>& col_data, NumMat<T>& row_data ) = 0; //与    host_generic_block_distributor.h

};//class BlockDistributorImpl

}//namespace detail

//接下来是BlockDistributor的实现

template <typename T>
BlockDistributor<T>::BlockDistributor( 
  std::unique_ptr<detail::BlockDistributorImpl<T>>&& impl 
) :
  impl_(std::move(impl)) { }                                   //unique_ptr实战！构造函数其一。

template <typename T>
BlockDistributor<T>::BlockDistributor() noexcept : impl_(nullptr) { };

//template <typename T>
//BlockDistributor<T>::BlockDistributor( MPI_Comm comm, Int M, Int N ) :
//  BlockDistributor( detail::make_default_host_<T>( comm, M, N ) ) { }
// 看样子曾经没有那个BlockDistributor_impl类？

template <typename T>
BlockDistributor<T>::~BlockDistributor() noexcept = default;   //构造函数其二


template <typename T>
BlockDistributor<T>& BlockDistributor<T>::operator=( BlockDistributor&& other ) noexcept {
  impl_ = std::move(other.impl_);
  return (*this);
}                                                              //赋值

template <typename T>
BlockDistributor<T>::BlockDistributor( BlockDistributor&& other ) noexcept :
  BlockDistributor( std::move(other.impl_) ) { }               //构造函数其三


template <typename T>
void BlockDistributor<T>::redistribute_row_to_col( const NumMat<T>& row_data, 
                                                         NumMat<T>& col_data ) {
  if( impl_ )//等价于impl_ != nullptr？
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

} // namespace scales
