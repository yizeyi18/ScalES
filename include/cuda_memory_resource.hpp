#pragma once
#include <experimental/memory_resource>

namespace cuda {

/**
 *  \brief Implementation of std::pmr::memory_resource which handles CUDA
 *  device memory management.
 */
class memory_resource : public std::experimental::pmr::memory_resource {

  using mr = std::experimental::pmr::memory_resource;

  /**
   *  @brief Allocate memory on the CUDA device
   *
   *  @param[in] bytes      Number of bytes to allocate
   *  @param[in] alignment  Requested alignment of allocated memory
   *
   *  @returns pointer to allocated memory
   */ 
  void* do_allocate(std::size_t bytes, std::size_t alignment);

  /**
   *  @brief Deallocate memory on the CUDA device
   *
   *  @param[in] p         Pointer to deallocate
   *  @param[in] bytes     Length of memory segment pointed to be p
   *  @param[in] alignment Alignment of p
   */ 
  void  do_deallocate(void * p, std::size_t bytes, std::size_t alignment);

  /**
   *  @brief Check if a std::pmr::memory_resource is a cuda::memory_resource
   *
   *  @param[in] other Memory resrouce to check
   *  @returns   true if other is castable to cuda::memory_resource, false
   *             otherwise.
   */ 
  bool do_is_equal( const mr& other ) const noexcept;

};

/**
 *  @brief Global accessor to an instance of cuda::memory_resource
 *
 *  @returns pointer to global instance of cuda::memory_resource
 */
memory_resource* get_default_resource();

}
