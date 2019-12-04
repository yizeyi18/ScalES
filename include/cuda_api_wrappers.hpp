#pragma once
#include <cstddef>
#include <type_traits>

namespace cuda {

namespace wrappers {

/**
 *  \brief Copy data from host to device
 *
 *  Copy data from host to device through CUDA API
 *  (cudaMemcpy)
 *
 *  @param[in/out] dest Device pointer to destination memory
 *  @param[in]     src  Host pointer to source memory
 *  @param[in]     len  Length of memory segment in bytes
 */
void memcpy_h2d( void* dest, const void* src, size_t len );

void memcpy2d_h2d( void* dest, size_t dpitch, const void* src, size_t spitch,
                   size_t width, size_t height );

/**
 *  \brief Copy data from device to host
 *
 *  Copy data from device to host through CUDA API
 *  (cudaMemcpy)
 *
 *  @param[in/out] dest Host pointer to destination memory
 *  @param[in]     src  Device pointer to source memory
 *  @param[in]     len  Length of memory segment in bytes
 */
void memcpy_d2h( void* dest, const void* src, size_t len );

void memcpy2d_d2h( void* dest, size_t dpitch, const void* src, size_t spitch,
                   size_t width, size_t height );

/**
 *  \brief Copy data on device
 *
 *  Copy data on device through CUDA API
 *  (cudaMemcpy)
 *
 *  @param[in/out] dest Device pointer to destination memory
 *  @param[in]     src  Device pointer to source memory
 *  @param[in]     len  Length of memory segment in bytes
 */
void memcpy_d2d( void* dest, const void* src, size_t len );

void memcpy2d_d2d( void* dest, size_t dpitch, const void* src, size_t spitch,
                   size_t width, size_t height );

/**
 *  \brief Allocate memory on the device
 *
 *  Allocate a memory segment on the device through the CUDA
 *  API (cudaMalloc)
 *
 *  @param[in] len Length of memory segment in bytes
 *  @returns   pointer to allocated memory segment
 */
void* malloc( size_t len );

/**
 *  \brief Allocate unified memory accessable from both
 *  host and device.
 *
 *  Allocate a segment of unified memory accessable from
 *  both host and device through the CUDA API 
 *  (cudaMallocManaged)
 *
 *  @param[in] len Length of memory segment in bytes
 *  @returns   pointer to allocated memory segment
 */
void* malloc_managed( size_t len );

/**
 *  \brief Allocate page locked memory on those host.
 *
 *  Allocate a page locked memory segment on the host
 *  through the CUDA API (cudaMallocHost). 
 *
 *  Yields better data movement performance in select 
 *  situations.
 *
 *  @param[in] len Length of memory segment in bytes
 *  @returns   pointer to allocated memory segment
 */
void* malloc_pinned( size_t len );


/**
 *  \brief Free (deallocate) a device memory segment
 *
 *  Deallocate a memory segment previously allocated by
 *  either malloc or malloc_managed (cudaFree).
 *
 *  @param[in] ptr Device pointer to memory segment to be
 *                 deallocated.
 */
void  free( void* ptr ); 

/**
 *  \brief Free (deallocate) a page locked host memory
 *  segment
 *
 *  Deallocate a memory segment previously allocated by
 *  malloc_pinned (cudaFreeHost).
 *
 *  @param[in] ptr Host pointer to memory segment to be
 *                 deallocated.
 */
void  free_pinned( void* ptr );

/**
 *  \brief Synchronize device and host
 *
 *   Synchronize the host and device through the CUDA API
 *   (cudaDeviceSynchronize). Host will wait until device
 *   has finished all enqueued operations.
 */
void device_sync();

/**
 *  \brief Initialize a raw device memory segment to an 
 *  integral value.
 *
 *  Initialize a raw device memory segment to an 
 *  integral value through the CUDA API (cudaMemset).
 *
 *  @param[in/out] data Device pointer of memory to manipulate
 *  @param[in]     val  Value to initialize elements of data
 *  @param[in]     len  Length of memory segment in bytes
 */
void memset( void* data, int val, size_t len );

}

/**
 *  \brief Copy data from host to device
 *
 *  Copy data from host to device through CUDA API
 *  (cudaMemcpy). Templated wrapper around 
 *  wrappers::memcpy_h2d. Data type must satisfy
 *  is_trivially_copyable
 *
 *  @tparam        T    Type of data to transfer
 *  @param[in/out] dest Device pointer to destination memory
 *  @param[in]     src  Host pointer to source memory
 *  @param[in]     len  Length of memory segment in bytes
 */
template <typename T>
std::enable_if_t< std::is_trivially_copyable<T>::value >
memcpy_h2d( T* dest, const T* src, size_t len ) {
  wrappers::memcpy_h2d( dest, src, len * sizeof(T) );
}

template <typename T>
std::enable_if_t< std::is_trivially_copyable<T>::value >
memcpy2d_h2d( T* dest, size_t dpitch, const T* src, size_t spitch,
              size_t width, size_t height ) {
  wrappers::memcpy2d_h2d( dest, dpitch * sizeof(T), src, spitch * sizeof(T),
                        width * sizeof(T), height );
}

/**
 *  \brief Copy data from device to host
 *
 *  Copy data from device to host through CUDA API
 *  (cudaMemcpy). Templated wrapper around
 *  wrappers::memcpy_d2h. Data type must satisfy
 *  is_trivially_copyable
 *
 *  @tparam        T    Type of data to transfer
 *  @param[in/out] dest Host pointer to destination memory
 *  @param[in]     src  Device pointer to source memory
 *  @param[in]     len  Length of memory segment in bytes
 */
template <typename T>
std::enable_if_t< std::is_trivially_copyable<T>::value >
memcpy_d2h( T* dest, const T* src, size_t len ) {
  wrappers::memcpy_d2h( dest, src, len * sizeof(T) );
}

template <typename T>
std::enable_if_t< std::is_trivially_copyable<T>::value >
memcpy2d_d2h( T* dest, size_t dpitch, const T* src, size_t spitch,
              size_t width, size_t height ) {
  wrappers::memcpy2d_d2h( dest, dpitch * sizeof(T), src, spitch * sizeof(T),
                        width * sizeof(T), height );
}

/**
 *  \brief Copy data on device
 *
 *  Copy data on devicethrough CUDA API
 *  (cudaMemcpy). Templated wrapper around
 *  wrappers::memcpy_d2h. Data type must satisfy
 *  is_trivially_copyable
 *
 *  @tparam        T    Type of data to transfer
 *  @param[in/out] dest Device pointer to destination memory
 *  @param[in]     src  Device pointer to source memory
 *  @param[in]     len  Length of memory segment in bytes
 */
template <typename T>
std::enable_if_t< std::is_trivially_copyable<T>::value >
memcpy_d2d( T* dest, const T* src, size_t len ) {
  wrappers::memcpy_d2d( dest, src, len * sizeof(T) );
}

template <typename T>
std::enable_if_t< std::is_trivially_copyable<T>::value >
memcpy2d_d2d( T* dest, size_t dpitch, const T* src, size_t spitch,
              size_t width, size_t height ) {
  wrappers::memcpy2d_d2d( dest, dpitch * sizeof(T), src, spitch * sizeof(T),
                        width * sizeof(T), height );
}

}
