#pragma once
#include "device_allocator.hpp"
#include "cuda_api_wrappers.hpp"
#include <vector>
#include <cassert>

namespace cuda {

template <typename T, typename Alloc = std::allocator<T>,
  typename = detail::enable_if_not_device_allocator_t<Alloc> >
using host_vector = std::vector<T, Alloc>;


template <typename T>
class device_vector : public std::vector<T, device_allocator<T>> {

  using base = std::vector<T, device_allocator<T>>;

public:

  using size_type      = typename base::size_type;
  using allocator_type = typename base::allocator_type;

  struct iterator : public base::iterator { 
    template <typename... Args>
    iterator( Args&&... args ) : base::iterator( std::forward<Args>(args)... ){};
  };

  using const_iterator = const iterator;

  // Inherit basic construction semantics from std::vector

  device_vector() : base() { }
  explicit device_vector( const allocator_type& alloc ) noexcept : base(alloc) { }

  explicit device_vector( size_type n, const T& val, 
    const allocator_type& alloc = allocator_type() ) : base( n, val, alloc ) { }

  explicit device_vector( size_type n, 
    const allocator_type& alloc = allocator_type() ) : base( n, alloc ) { }

  explicit device_vector( device_vector&& other ) : base( std::move(other) ) { }


  // Copy semantics are special

  explicit device_vector( const device_vector& other ) : 
    device_vector( other.size() ) /* Allocate device memory */ {

    if( this->size() )
      memcpy_d2d( this->data(), other.data(), this->size() );

  }

  template <typename Alloc>
  explicit device_vector( const host_vector<T, Alloc>& other ) :
    device_vector(other.size()) /* Allocate device memory */ {

    // Copy to device
    if( this->size() )
      memcpy_h2d( this->data(), other.data(), this->size() ); 

  }


  iterator begin(){ return iterator( this->data() ); }
  iterator end()  { return iterator( this->data() + this->size() ); }
};



template <typename T>
using managed_vector = std::vector<T, managed_allocator<T>>;

template <typename T>
using pinned_vector = std::vector<T, pinned_allocator<T>>;



template <typename VectorDevice, typename VectorHost>
std::enable_if_t<
  std::is_same_v< typename VectorDevice::value_type, typename VectorHost::value_type > 
  and 
      detail::is_device_allocator_v<typename VectorDevice::allocator_type> and
  not detail::is_device_allocator_v<typename VectorHost::allocator_type>
>
copy( const VectorDevice& v_d, VectorHost& v_h ) {
  assert( v_d.size() == v_h.size() );
  memcpy_d2h( v_h.data(), v_d.data(), v_d.size() );
}

template <typename VectorDevice, typename VectorHost>
std::enable_if_t<
  std::is_same_v< typename VectorDevice::value_type, typename VectorHost::value_type > 
  and 
      detail::is_device_allocator_v<typename VectorDevice::allocator_type> and
  not detail::is_device_allocator_v<typename VectorHost::allocator_type>
>
copy( const VectorHost& v_h, VectorDevice& v_d ) {
  assert( v_d.size() == v_h.size() );
  memcpy_h2d( v_d.data(), v_h.data(), v_d.size() );
}

template <typename VectorDevice>
std::enable_if_t<
  detail::is_device_allocator_v<typename VectorDevice::allocator_type>
>
copy( const VectorDevice& v_d, VectorDevice& v_d_src ) {
  assert( v_d.size() == v_d_src.size() );
  memcpy_d2d( v_d.data(), v_d_src.data(), v_d.size() );
}

}
