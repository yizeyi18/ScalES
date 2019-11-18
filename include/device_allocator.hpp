#include <type_traits>
#include <cuda_api_wrappers.hpp>

namespace cuda {

namespace detail {

template <typename T>
struct no_construct_allocator_base {

  using value_type = T;

  template <typename... Args>
  constexpr void construct( T* ptr, Args&&... args ){ };

  constexpr void destroy( T* ptr ){ }

  no_construct_allocator_base( ) noexcept                         = default;
  no_construct_allocator_base( const no_construct_allocator_base& ) noexcept = default;
  no_construct_allocator_base( no_construct_allocator_base&& ) noexcept      = default;
  ~no_construct_allocator_base( ) noexcept                        = default;

};

template <typename T, typename U = void>
using enable_if_trivially_copyable_t = 
  typename std::enable_if< std::is_trivially_copyable_v<T>, U>::type;

}



template <typename T, typename = detail::enable_if_trivially_copyable_t<T> >
struct device_allocator : public detail::no_construct_allocator_base<T> {

  using value_type = typename detail::no_construct_allocator_base<T>::value_type;

  T* allocate( size_t n ) {
    return (T*)wrappers::malloc( n * sizeof(T) );
  }
  void deallocate( T* ptr, size_t n ) {
    wrappers::free( (void*)ptr );
  }

  device_allocator( ) noexcept                         = default;
  device_allocator( const device_allocator& ) noexcept = default;
  device_allocator( device_allocator&& ) noexcept      = default;
  ~device_allocator( ) noexcept                        = default;

};


template <typename T, typename = detail::enable_if_trivially_copyable_t<T> >
struct managed_allocator : public detail::no_construct_allocator_base<T> {

  using value_type = typename detail::no_construct_allocator_base<T>::value_type;

  T* allocate( size_t n ) {
    return (T*)wrappers::malloc_managed( n * sizeof(T) );
  }
  void deallocate( T* ptr, size_t n ) {
    wrappers::free( (void*)ptr );
  }


  managed_allocator( ) noexcept                         = default;
  managed_allocator( const managed_allocator& ) noexcept = default;
  managed_allocator( managed_allocator&& ) noexcept      = default;
  ~managed_allocator( ) noexcept                        = default;

};





template <typename T>
struct pinned_allocator {

  using value_type      = T;

  T* allocate( size_t n ) {
    return (T*)wrappers::malloc_pinned( n * sizeof(T) );
  }
  void deallocate( T* ptr, size_t n ) {
    wrappers::free_pinned( (void*)ptr );
  }

  // Construct and Destroy get handled by allocator_traits

  pinned_allocator( ) noexcept                         = default;
  pinned_allocator( const pinned_allocator& ) noexcept = default;
  pinned_allocator( pinned_allocator&& ) noexcept      = default;
  ~pinned_allocator( ) noexcept                        = default;

};





namespace detail {

template <typename Alloc>
struct is_device_allocator : public std::false_type { };

template <typename T>
struct is_device_allocator< device_allocator<T> > : public std::true_type { };
template <typename T>
struct is_device_allocator< managed_allocator<T> > : public std::true_type { };

template <typename Alloc>
inline constexpr bool is_device_allocator_v = is_device_allocator<Alloc>::value;

template <typename Alloc, typename U = void>
using enable_if_device_allocator_t = 
  typename std::enable_if< is_device_allocator_v<Alloc>, U >::type;

template <typename Alloc, typename U = void>
using enable_if_not_device_allocator_t = 
  typename std::enable_if< not is_device_allocator_v<Alloc>, U >::type;

}




}
