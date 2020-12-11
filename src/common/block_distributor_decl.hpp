#pragma once

#include "environment.hpp"
#include "nummat_decl.hpp"
#include "numvec_decl.hpp"
#include <memory>

namespace dgdft {
namespace dist_util {

namespace detail {

template <typename T>
class BlockDistributorImpl;


template <typename T>
std::unique_ptr< BlockDistributorImpl<T> > 
  make_default_host_distributor( MPI_Comm comm, Int M, Int N );

#ifdef DEVICE 
template <typename T>
std::unique_ptr< BlockDistributorImpl<T> > 
  make_default_device_distributor( MPI_Comm comm, Int M, Int N );
#endif

}

template <typename T>
class BlockDistributor {

  std::unique_ptr< detail::BlockDistributorImpl<T> > impl_;

public:

  BlockDistributor() noexcept;
  BlockDistributor( std::unique_ptr< detail::BlockDistributorImpl<T> >&& impl );
  BlockDistributor( MPI_Comm comm, Int M, Int N );
  ~BlockDistributor() noexcept;

  void redistribute_row_to_col( const NumMat<T>& row_data, NumMat<T>& col_data );
  void redistribute_col_to_row( const NumMat<T>& col_data, NumMat<T>& row_data );

#if 1 // These should not be needed in production

  const IntNumVec* sendcounts() const;
  const IntNumVec* recvcounts() const;
  const IntNumVec* senddispls() const;
  const IntNumVec* recvdispls() const;
  const IntNumMat* sendk()      const;
  const IntNumMat* recvk()      const;

#endif
};

} // namespace dist_util
} // namespace dgdft
