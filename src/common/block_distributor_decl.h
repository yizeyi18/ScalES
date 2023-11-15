//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: David Williams-Young

/// 怎么这个用CPP特性用得这么多？
/// 可能是新时代CPP的标准操作吧。
//  yizeyi18 2023.11.14
#pragma once

#include "environment.h"
#include "nummat_decl.h"
#include "numvec_decl.h"
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

}//detail

//所以为什么没把这个拿unique_ptr写？
template <typename T>
class BlockDistributor {
  //CPP小课堂：std::unique_ptr在构造·赋值时会破旧立新
  //换言之没有两个合法使用的std::unique_ptr指向同一对象
  std::unique_ptr< detail::BlockDistributorImpl<T> > impl_;

public:
  //三种构造函数，无参析构函数，以及想用std::unique_ptr但还是没用于是被迫手动删除的拷贝构造函数
  //CPP小课堂：显式声明noexcept后，逢error必报错退出，外头有catch也接不着
  BlockDistributor() noexcept;
  BlockDistributor( std::unique_ptr< detail::BlockDistributorImpl<T> >&& impl );
  //这个用不用标注为delete？
  //BlockDistributor( MPI_Comm comm, Int M, Int N );
  ~BlockDistributor() noexcept;

  BlockDistributor( const BlockDistributor& ) = delete;
  BlockDistributor( BlockDistributor&& ) noexcept;

  BlockDistributor& operator=( const BlockDistributor& ) = delete;
  BlockDistributor& operator=( BlockDistributor&& ) noexcept;

  void redistribute_row_to_col( const NumMat<T>& row_data, NumMat<T>& col_data );
  void redistribute_col_to_row( const NumMat<T>& col_data, NumMat<T>& row_data );

};//class BlockDistributor


template <typename T>
BlockDistributor<T> make_block_distributor( BlockDistAlg alg, MPI_Comm comm, Int M, Int N );

} // namespace scales
