#include <cuda_memory_resource.hpp>

namespace cuda {

template <typename T>
class no_init_allocator : public std::experimental::pmr::polymorphic_allocator<T> {

  using backend = std::experimental::pmr::polymorphic_allocator<T>;
  using mr      = std::experimental::pmr::memory_resource;

public:
  
  no_init_allocator() noexcept : backend() { };
  no_init_allocator( const backend& other ) : backend(other) { };
  no_init_allocator( mr* r) : backend( r ) { };

  template <typename... Args>
  void construct( T* p, Args&&... args ){ }

  void destroy( T* p ){ }

};


template <typename T>
class device_allocator : public no_init_allocator<T> {

  using backend = no_init_allocator<T>;
  using mr      = std::experimental::pmr::memory_resource;

public:

  device_allocator() noexcept : backend(cuda::get_default_resource()) { };
  device_allocator( const backend& other ) : backend(other) { };
  //device_allocator( mr* r) = delete;
  device_allocator( mr* r) : backend(r){ };


};

}
