#pragma once
#include <list>
#if __has_include(<memory_resource>)
  #include <memory_resource>
#else
  #include <experimental/memory_resource>
#endif

namespace memory {

namespace detail {

constexpr inline int64_t bytes_distance( 
  void* first, void* last ) {
  return (char*)last - (char*)first;
}

inline size_t div_ceil( size_t n, size_t m ) {
  auto div = std::div( (long long)n, (long long)m );
  return div.quot + !!div.rem;
}

}

#if __has_include(<memory_resource>)
using memory_resource = std::pmr::memory_resource;
#else
using memory_resource = std::experimental::pmr::memory_resource;
#endif

class segregated_memory_resource : public memory_resource {

  struct memory_block {
    void* top;
    std::size_t nbytes;
    std::size_t block_size;
    std::size_t nblocks;
  };
  
  struct memory_chunk {
    std::list< memory_block >::iterator parent_block;
    void* ptr;
  };
  
  memory_resource* upstream_ = nullptr;

  std::list< memory_block > mem_blocks_;
  std::list< memory_chunk > free_list_;
  std::size_t               block_size_;


  void* do_allocate( std::size_t, std::size_t );
  void  do_deallocate( void*, std::size_t, std::size_t );
  bool  do_is_equal( const memory_resource& ) const noexcept ;


  void add_block( std::list< memory_block >::iterator );

public:

  segregated_memory_resource() = delete;
  segregated_memory_resource( std::size_t, std::size_t, 
    memory_resource* );
  segregated_memory_resource( std::size_t, std::size_t );

  ~segregated_memory_resource() noexcept;

  segregated_memory_resource( const segregated_memory_resource& ) = delete;
  segregated_memory_resource( segregated_memory_resource&& ) noexcept;


  decltype(mem_blocks_) mem_blocks() const noexcept;
  decltype(free_list_)  free_list()  const noexcept;
  std::size_t           block_size() const noexcept;

}; // segregated_memory_resource

}
