#pragma once
#include "host_generic_block_distributor.hpp"
#include "host_optpack_block_distributor.hpp"

namespace dgdft {
namespace dist_util {


namespace detail {


template <typename T>
std::unique_ptr< BlockDistributorImpl<T> > 
  make_default_host_distributor( MPI_Comm comm, Int M, Int N ) {

  return std::unique_ptr< BlockDistributorImpl<T> >(
    new HostOptPackBlockDistributor<T>( comm, M, N )
  );

}

}

}
}
