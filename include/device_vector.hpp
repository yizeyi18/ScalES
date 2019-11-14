#include <device_allocator.hpp>
#include <vector>

namespace cuda {

template <typename T>
  using device_vector = std::vector< T, cuda::device_allocator<T> >;

/*
template <typename T>
class device_vector : public std::vector< T, cuda::device_allocator<T> > {

  using base = std::vector< T, cuda::device_allocator<T> >;

public:

  //using value_type      = typename base::value_type;
  //using allocator_type  = typename base::allocator_type;
  //using size_type       = typename base::size_type;
  //using difference_type = typename base::difference_type;

};
*/

}
