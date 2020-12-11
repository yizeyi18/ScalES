#include "host_block_distributor.hpp"

namespace dgdft {
namespace dist_util {

template class HostBlockDistributor<double>;


namespace detail {

template 
std::unique_ptr< BlockDistributorImpl<double> > 
  make_default_host_distributor<double>( MPI_Comm comm, Int M, Int N );

}

}
}
