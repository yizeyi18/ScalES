#include <cuda_api_wrappers.hpp>
#include <exceptions.hpp>
#include <iostream>

namespace cuda {
namespace wrappers {

void memset( void* data, int val, size_t len ) {
  CUDA_THROW( cudaMemset( data, val, len ) );
}

void device_sync() {
  CUDA_THROW( cudaDeviceSynchronize() );
}

void memcpy_h2d( void* dest, const void* src, size_t len ) {
  CUDA_THROW( cudaMemcpy( dest, src, len, cudaMemcpyHostToDevice ) );
}

void memcpy_d2h( void* dest, const void* src, size_t len ) {
  CUDA_THROW( cudaMemcpy( dest, src, len, cudaMemcpyDeviceToHost ) );
}

void memcpy_d2d( void* dest, const void* src, size_t len ) {
  CUDA_THROW( cudaMemcpy( dest, src, len, cudaMemcpyDeviceToDevice ) );
}

void* malloc( size_t len ) {

  void* ptr;
  CUDA_THROW( cudaMalloc( &ptr, len ) );
  //std::cout << "CUDA MALLOC " << len << ", " << ptr << std::endl;
  return ptr;

}

void* malloc_pinned( size_t len ) {

  void* ptr;
  CUDA_THROW( cudaMallocHost( &ptr, len ) );
  //std::cout << "CUDA MALLOC HOST" << len << ", " << ptr << std::endl;
  return ptr;

}

void* malloc_managed( size_t len ) {

  void* ptr;
  CUDA_THROW( cudaMallocManaged( &ptr, len ) );
  //std::cout << "CUDA MALLOC HOST" << len << ", " << ptr << std::endl;
  return ptr;

}

void  free( void* ptr ) {
  //std::cout << "CUDA FREE " << ptr << std::endl;
  CUDA_THROW( cudaFree( ptr ) );
}

void  free_pinned( void* ptr ) {
  //std::cout << "CUDA FREE Host" << ptr << std::endl;
  CUDA_THROW( cudaFreeHost( ptr ) );
}


}
}

