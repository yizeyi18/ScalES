#pragma once

#include "block_distributor_impl.hpp"
#include "host_generic_block_distributor.hpp"
#include "host_optpack_block_distributor.hpp"


namespace scales {

namespace detail {

template <typename T, class ImplType, typename... Args>
std::unique_ptr< BlockDistributorImpl<T> >
  make_distributor_impl( Args&&... args ) {

  return std::unique_ptr< BlockDistributorImpl<T> >(
    new ImplType( std::forward<Args>(args)... )
  );

}

template <typename T, class ImplType, typename... Args>
BlockDistributor<T> make_distributor( Args&&... args ) {

  return BlockDistributor<T>( 
    make_distributor_impl<T,ImplType>( std::forward<Args>(args)... )
  );

}

}

template <typename T>
BlockDistributor<T> make_block_distributor( BlockDistAlg alg, 
                                            MPI_Comm comm, 
                                            Int M, 
                                            Int N ) {

  switch(alg) {
    case BlockDistAlg::HostGeneric:
      return detail::make_distributor<T,HostGenericBlockDistributor<T>>(
        comm, M, N
      );
    case BlockDistAlg::HostOptPack:
      return detail::make_distributor<T,HostOptPackBlockDistributor<T>>(
        comm, M, N
      );
    default:
      return BlockDistributor<T>();
  }

}

}
