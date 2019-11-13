#include <cuda_memory_resource.hpp>
#include <cuda_wrapper.hpp>
#include <iostream>

namespace cuda {

void* memory_resource::do_allocate( std::size_t bytes, std::size_t alignment ) {
  return cuda::malloc( bytes );
}

void memory_resource::do_deallocate( void* p, std::size_t bytes, std::size_t alignment ) {

  cuda::free( p );

}

using mr = std::experimental::pmr::memory_resource;
bool memory_resource::do_is_equal( const mr& other ) const noexcept {

  bool is_eq = false;
  try {
    const memory_resource& temp = dynamic_cast< const memory_resource& >(other);
    is_eq = true;
  } catch( const std::bad_cast& ){ 
    is_eq = false;
  } 

  return is_eq;

}



memory_resource default_cuda_resource;
memory_resource* get_default_resource(){ return &default_cuda_resource; }

}
