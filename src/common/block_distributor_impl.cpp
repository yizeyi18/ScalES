#include "block_distributor_impl.hpp"
#include "block_distributor_factory.hpp"

namespace dgdft {

#define bdist_impl(T) \
template class BlockDistributor<T>; \
template BlockDistributor<T> make_block_distributor( BlockDistAlg alg,\
                                            MPI_Comm comm, \
                                            Int M, \
                                            Int N );


bdist_impl(double);
// TODO Complex


} // namespace dgdft
